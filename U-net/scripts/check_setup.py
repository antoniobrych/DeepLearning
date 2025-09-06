from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from typing import Union


def add_src_to_path() -> None:
    
    this_dir = Path(__file__).resolve().parent
    u_net_root = this_dir.parent
    src_path = u_net_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


add_src_to_path()

from data import CamVidDataset 
from models.unet import Unet 


def center_crop_2d(mask: Tensor, target_h: int, target_w: int) -> Tensor:
    """Center-crop a 2D or 3D tensor to (target_h, target_w).

    Accepts tensors shaped [H, W] or [N, H, W].
    """
    if mask.dim() == 2:
        h, w = mask.shape
        dh = (h - target_h) // 2
        dw = (w - target_w) // 2
        return mask[dh : dh + target_h, dw : dw + target_w]
    elif mask.dim() == 3:
        n, h, w = mask.shape
        dh = (h - target_h) // 2
        dw = (w - target_w) // 2
        return mask[:, dh : dh + target_h, dw : dw + target_w]
    else:
        raise ValueError(f"Unexpected mask shape {tuple(mask.shape)}")


def infer_num_classes(colors_csv: Path) -> int:
    import pandas as pd

    df = pd.read_csv(colors_csv)
    return len(df)


from typing import Union

def load_split(root: Path, split: str, ignore_index: Union[int, None] = None) -> CamVidDataset:
    images = root / split
    labels = root / f"{split}_labels"
    colors_csv = root / "class_dict.csv"
    assert images.exists() and labels.exists(), f"Missing split folders for {split} under {root}"
    ds = CamVidDataset(
        image_dir=images,
        mask_dir=labels,
        colors_csv=colors_csv,
        ignore_index=ignore_index,
    )
    return ds


def check_batch_shapes(
    images: Tensor, masks: Tensor, num_classes: int, depth: int = 4, in_channels: int = 3, width: int = 64
) -> Tuple[Tensor, Tensor, Tensor]:
    """Run a forward pass and compute a dummy loss to validate pipeline.

    Returns (logits, cropped_masks, loss)
    """
    model = Unet(num_classes=num_classes, depth=depth, in_channels=in_channels, width=width)
    model.eval()

    with torch.inference_mode():
        logits: Tensor = model(images)

    _, _, H_out, W_out = logits.shape
    masks_c = center_crop_2d(masks, H_out, W_out)

    assert images.dtype == torch.float32, f"Images dtype {images.dtype} != float32"
    assert masks.dtype == torch.long, f"Masks dtype {masks.dtype} != long"
    assert logits.shape[0] == images.shape[0], "Batch size mismatch"
    assert logits.shape[1] == num_classes, f"num_classes mismatch: {logits.shape[1]} != {num_classes}"

    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(logits, masks_c)
    return logits, masks_c, loss


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "data" / "CamVid"
    assert root.exists(), f"CamVid root not found: {root}"

    train_ds = load_split(root, "train")
    val_ds = load_split(root, "val")
    test_ds = load_split(root, "test")

    print(f"Train samples: {len(train_ds)}; Val: {len(val_ds)}; Test: {len(test_ds)}")

    num_classes = infer_num_classes(root / "class_dict.csv")
    print(f"Detected classes: {num_classes}")

    loader = DataLoader(train_ds, batch_size=2, shuffle=False)
    images, masks = next(iter(loader))
    print(f"Batch shapes -> images: {tuple(images.shape)}, masks: {tuple(masks.shape)}")

    logits, masks_c, loss = check_batch_shapes(images, masks, num_classes=num_classes, depth=4)
    print(f"Logits shape: {tuple(logits.shape)}; Cropped masks: {tuple(masks_c.shape)}")
    print(f"Dummy CE loss: {loss.item():.4f}")

    print("All checks passed.")


if __name__ == "__main__":
    main()

