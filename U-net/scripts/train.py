import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from pathlib import Path
import sys
import csv
from check_setup import *

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from data import CamVidDataset
from models.unet import Unet

def compute_confusion_matrix(preds, targets, num_classes):
    preds = preds.view(-1)
    targets = targets.view(-1)
    mask = (targets >= 0) & (targets < num_classes)
    hist = torch.bincount(
        num_classes * targets[mask] + preds[mask],
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes).to(torch.float32)
    return hist


def compute_metrics(conf_matrix):
    tp = torch.diag(conf_matrix)
    fp = conf_matrix.sum(dim=0) - tp
    fn = conf_matrix.sum(dim=1) - tp

    pixel_acc = tp.sum() / conf_matrix.sum()
    iou_per_class = tp / (tp + fp + fn + 1e-7)
    mean_iou = iou_per_class.mean()
    dice_per_class = 2 * tp / (2 * tp + fp + fn + 1e-7)
    mean_dice = dice_per_class.mean()

    return {
        "PixelAcc": pixel_acc.item(),
        "MeanIoU": mean_iou.item(),
        "MeanDice": mean_dice.item(),
        "F1_macro": mean_dice.item()
    }

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        targets_onehot = torch.nn.functional.one_hot(
            targets, num_classes=logits.shape[1]
        ).permute(0,3,1,2).float()

        intersection = (probs * targets_onehot).sum(dim=(2,3))
        union = probs.sum(dim=(2,3)) + targets_onehot.sum(dim=(2,3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class GeneralizedCrossEntropy(nn.Module):
    def __init__(self, q=0.7):
        super().__init__()
        self.q = q

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = (1 - pt**self.q) / self.q
        return loss.mean()

class SymmetricCrossEntropy(nn.Module):
    def __init__(self, alpha=0.1, beta=1.0, num_classes=32):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        probs = torch.softmax(logits, dim=1)
        one_hot = torch.nn.functional.one_hot(targets, self.num_classes).float()
        rce = (-torch.sum(probs * torch.log(one_hot + 1e-7), dim=1)).mean()
        return self.alpha * ce_loss + self.beta * rce

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
def validate(model, loader, loss_fn, device, num_classes):
    model.eval()
    total_loss = 0.0
    conf_matrix = torch.zeros((num_classes, num_classes), dtype=torch.float64)

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        _, _, H_out, W_out = logits.shape
        masks_c = center_crop_2d(masks, H_out, W_out)

        loss = loss_fn(logits, masks_c)
        total_loss += loss.item() * images.size(0)

        preds = torch.argmax(logits, dim=1)
        conf_matrix += compute_confusion_matrix(preds.cpu(), masks_c.cpu(), num_classes)

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(conf_matrix)
    return avg_loss, metrics

def main_train():
    root = Path(__file__).resolve().parents[1] / "data" / "CamVid"
    num_classes = infer_num_classes(root / "class_dict.csv")

    train_ds = load_split(root, "train")
    val_ds = load_split(root, "val")

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print("Using device:", device)

    optimizers_to_try = {
        "Adam": lambda params: optim.Adam(params, lr=1e-3),
        "SGD": lambda params: optim.SGD(params, lr=1e-2, momentum=0.9),
        "RMSprop": lambda params: optim.RMSprop(params, lr=1e-3, momentum=0.9),
        "AdamW": lambda params: optim.AdamW(params, lr=1e-3)
    }

    losses_to_try = {
        "CrossEntropy": nn.CrossEntropyLoss(),
        "DiceLoss": DiceLoss(),
        "GCE": GeneralizedCrossEntropy(q=0.7),
        "SCE": SymmetricCrossEntropy(alpha=0.1, beta=1.0, num_classes=num_classes),
        "CE+Dice": (nn.CrossEntropyLoss(), DiceLoss())
    }

    num_epochs = 2
    csv_file = "results_metrics.csv"

    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        header = ["Optimizer", "Loss", "Epoch", "TrainLoss", "ValLoss", "PixelAcc", "MeanIoU", "MeanDice", "F1_macro"]
        writer.writerow(header)

        for opt_name, opt_fn in optimizers_to_try.items():
            for loss_name, loss_obj in losses_to_try.items():
                print(f"\n=== Treinando com {opt_name} + {loss_name} ===")

                model = Unet(num_classes=num_classes, depth=4, in_channels=3, width=64).to(device)
                optimizer = opt_fn(model.parameters())

                for epoch in range(1, num_epochs + 1):
                    if isinstance(loss_obj, tuple):
                        ce_loss, dice_loss = loss_obj
                        def combined_loss(logits, masks):
                            return ce_loss(logits, masks) + dice_loss(logits, masks)
                        train_loss = train_one_epoch(model, train_loader, optimizer, combined_loss, device)
                        val_loss, val_metrics = validate(model, val_loader, combined_loss, device, num_classes)
                    else:
                        train_loss = train_one_epoch(model, train_loader, optimizer, loss_obj, device)
                        val_loss, val_metrics = validate(model, val_loader, loss_obj, device, num_classes)

                    print(f"[{opt_name}+{loss_name}] Epoch {epoch:02d} | "
                          f"Train {train_loss:.4f} | Val {val_loss:.4f} | "
                          f"IoU {val_metrics['MeanIoU']:.4f} | "
                          f"Acc {val_metrics['PixelAcc']:.4f} | "
                          f"Dice {val_metrics['MeanDice']:.4f}")

                    row = [
                        opt_name, loss_name, epoch, train_loss, val_loss,
                        val_metrics["PixelAcc"], val_metrics["MeanIoU"],
                        val_metrics["MeanDice"], val_metrics["F1_macro"]
                    ]
                    writer.writerow(row)


if __name__ == "__main__":
    main_train()
