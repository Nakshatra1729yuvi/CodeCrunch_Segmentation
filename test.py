# test.py
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import losses as smp_losses
import argparse

def main(args):
    DATA_DIR = args.data_dir
    TEST_DIR = os.path.join(DATA_DIR, 'test')
    MODEL_PATH = args.model_path
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

    val_transform = A.Compose([
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.Normalize(),
        ToTensorV2()
    ])

    test_ds = SegDataset(TEST_DIR, transform=val_transform)

    test_loader = DataLoader(
        test_ds,
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
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(DEVICE)
    model.eval()

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

    test_losses = []
    test_ious = []
    test_accs = []

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            # Original prediction
            outputs = model(images)

            # TTA: Horizontal flip
            images_flip = torch.flip(images, dims=[3])
            outputs_flip = model(images_flip)
            outputs_flip = torch.flip(outputs_flip, dims=[3])

            # Average logits
            avg_logits = (outputs + outputs_flip) / 2
            probs = torch.softmax(avg_logits, dim=1)

            loss = combined_loss(avg_logits, masks)
            test_losses.append(loss.item())

            iou = compute_iou(avg_logits, masks, NUM_CLASSES)
            test_ious.append(iou)

            pred = torch.argmax(avg_logits, dim=1)
            acc = (pred == masks).float().mean().item()
            test_accs.append(acc)

    mean_test_loss = np.mean(test_losses)
    mean_test_iou = np.mean(test_ious)
    mean_test_acc = np.mean(test_accs)

    print(f"Mean Test Loss: {mean_test_loss:.4f}")
    print(f"Mean Test IoU: {mean_test_iou:.4f}")
    print(f"Mean Test Pixel Accuracy: {mean_test_acc:.4f}")

    with open(os.path.join(OUTPUT_DIR, 'test_metrics.txt'), 'w') as f:
        f.write(f"Mean Test Loss: {mean_test_loss:.4f}\n")
        f.write(f"Mean Test IoU: {mean_test_iou:.4f}\n")
        f.write(f"Mean Test Pixel Accuracy: {mean_test_acc:.4f}\n")

    print("Test metrics saved to test_metrics.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the segmentation model")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory containing train/val/test")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save the test metrics")
    args = parser.parse_args()
    main(args)