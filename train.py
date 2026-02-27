import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm
import matplotlib.pyplot as plt
from segmentation_models_pytorch import losses as smp_losses
import argparse

def main(args):
    DATA_DIR = args.data_dir
    TRAIN_DIR = os.path.join(DATA_DIR, 'train')
    VAL_DIR = os.path.join(DATA_DIR, 'val')
    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    value_map = {
        0: 0,
        100: 1,
        200: 2,
        300: 3,
        500: 4,
        550: 5,
        700: 6,
        800: 7,
        7100: 8,
        10000: 9
    }
    NUM_CLASSES = len(value_map)

    class SegDataset(Dataset):
        def __init__(self, root_dir, transform=None):
            self.image_dir = os.path.join(root_dir, "Color_Images")
            self.mask_dir = os.path.join(root_dir, "Segmentation")
            self.images = sorted(os.listdir(self.image_dir))
            self.transform = transform

        def convert_mask(self, mask):
            mask_new = np.zeros_like(mask)
            for raw_val, new_val in value_map.items():
                mask_new[mask == raw_val] = new_val
            return mask_new

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            img_path = os.path.join(self.image_dir, self.images[idx])
            mask_path = os.path.join(self.mask_dir, self.images[idx])

            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            mask = self.convert_mask(mask)

            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]

            return image, mask.long()

    IMAGE_SIZE = 512

    train_transform = A.Compose([
        A.RandomResizedCrop(size=(IMAGE_SIZE, IMAGE_SIZE), scale=(0.6, 1.0)),
        
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.2),
        
        A.OneOf([
            A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.15),
            A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5),
        ], p=0.7),

        A.RandomGamma(gamma_limit=(70, 130), p=0.5),

        A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
        
        A.MotionBlur(blur_limit=5, p=0.3),

        A.CoarseDropout(max_holes=8, max_height=64, max_width=64, p=0.4),

        A.ToGray(p=0.3),

        A.Normalize(),
        ToTensorV2()
    ])
    val_transform = A.Compose([
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.Normalize(),
        ToTensorV2()
    ])

    train_ds = SegDataset(TRAIN_DIR, transform=train_transform)
    val_ds = SegDataset(VAL_DIR, transform=val_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=8,
        shuffle=True,
        num_workers=2,
        drop_last=True 
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=8,
        shuffle=False,
        num_workers=2,
        drop_last=False
    )

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", DEVICE)

    model = smp.DeepLabV3Plus(
        encoder_name="mit_b3",
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES
    )

    model.to(DEVICE)

    class DiceLoss(nn.Module):
        def __init__(self, smooth=1e-6):
            super().__init__()
            self.smooth = smooth

        def forward(self, preds, targets):
            preds = torch.softmax(preds, dim=1)
            targets_one_hot = torch.nn.functional.one_hot(targets, NUM_CLASSES)
            targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

            intersection = (preds * targets_one_hot).sum(dim=(2,3))
            union = preds.sum(dim=(2,3)) + targets_one_hot.sum(dim=(2,3))

            dice = (2 * intersection + self.smooth) / (union + self.smooth)
            return 1 - dice.mean()

    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss()

    lovasz_loss = smp_losses.LovaszLoss(mode='multiclass', per_image=False)

    def combined_loss(pred, target):
        return 0.4 * ce_loss(pred, target) + \
               0.3 * dice_loss(pred, target) + \
               0.3 * lovasz_loss(pred, target)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
    scaler = torch.amp.GradScaler("cuda")

    def compute_iou(pred, target, num_classes):
        pred = torch.argmax(pred, dim=1)
        ious = []

        for cls in range(num_classes):
            pred_inds = (pred == cls)
            target_inds = (target == cls)

            intersection = (pred_inds & target_inds).sum().float()
            union = (pred_inds | target_inds).sum().float()

            if union == 0:
                ious.append(np.nan)
            else:
                ious.append((intersection / union).item())

        return np.nanmean(ious)

    EPOCHS = 10
    best_iou = 0

    train_losses = []
    train_ious = []
    train_accs = []
    val_losses = []
    val_ious = []
    val_accs = []

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        train_iou = 0
        train_acc = 0
        num_batches = len(train_loader)

        for images, masks in tqdm(train_loader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = combined_loss(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            iou = compute_iou(outputs, masks, NUM_CLASSES)
            train_iou += iou

            pred = torch.argmax(outputs, dim=1)
            acc = (pred == masks).float().mean().item()
            train_acc += acc

        train_loss /= num_batches
        train_iou /= num_batches
        train_acc /= num_batches

        train_losses.append(train_loss)
        train_ious.append(train_iou)
        train_accs.append(train_acc)

        scheduler.step()

        model.eval()
        val_loss = 0
        val_iou = 0
        val_acc = 0
        num_batches = len(val_loader)

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)

                outputs = model(images)
                loss = combined_loss(outputs, masks)
                val_loss += loss.item()

                iou = compute_iou(outputs, masks, NUM_CLASSES)
                val_iou += iou

                pred = torch.argmax(outputs, dim=1)
                acc = (pred == masks).float().mean().item()
                val_acc += acc

        val_loss /= num_batches
        val_iou /= num_batches
        val_acc /= num_batches

        val_losses.append(val_loss)
        val_ious.append(val_iou)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Train IoU: {train_iou:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f} | Val Acc: {val_acc:.4f}")

        if val_iou >= best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))

    print("Best Val IoU:", best_iou)

    epochs = range(1, EPOCHS + 1)

    # Loss plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'loss_plot.png'))
    plt.close()

    # IoU plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_ious, label='Train IoU')
    plt.plot(epochs, val_ious, label='Val IoU')
    plt.title('IoU over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'iou_plot.png'))
    plt.close()

    # Pixel Accuracy plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_accs, label='Train Pixel Accuracy')
    plt.plot(epochs, val_accs, label='Val Pixel Accuracy')
    plt.title('Pixel Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Pixel Accuracy')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'acc_plot.png'))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the segmentation model")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory containing train/val/test")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save the model and plots")
    args = parser.parse_args()
    main(args)