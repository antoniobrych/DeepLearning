import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from pathlib import Path
import sys
from pathlib import Path
from check_setup import *
ROOT = Path(__file__).resolve().parent.parent
print(ROOT)
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from data import CamVidDataset
from models.unet import Unet


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)

        logits = model(images)
        _, _, H_out, W_out = logits.shape
        masks_c = center_crop_2d(masks, H_out, W_out)

        loss = loss_fn(logits, masks_c)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        _, _, H_out, W_out = logits.shape
        masks_c = center_crop_2d(masks, H_out, W_out)

        loss = loss_fn(logits, masks_c)
        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def main_train():
    root = Path(__file__).resolve().parents[1] / "data" / "CamVid"
    num_classes = infer_num_classes(root / "class_dict.csv")

    train_ds = load_split(root, "train")
    val_ds = load_split(root, "val")

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    print("Using device:", device)

    model = Unet(num_classes=num_classes, depth=4, in_channels=3, width=64).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    num_epochs = 1
    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = validate(model, val_loader, loss_fn, device)

        print(f"Epoch {epoch:02d} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

    torch.save(model.state_dict(), "unet_camvid.pth")
    print("Training complete. Model saved.")


if __name__ == "__main__":
    main_train()
