# Offroad Segmentation Model

## Overview
This repository contains scripts to train and test a semantic segmentation model for offroad environments using the DeepLabV3+ architecture with a MiT-B3 backbone. The model is trained on the Duality Offroad Segmentation dataset.

## Requirements
- Python 3.8 or higher
- GPU recommended for training (CUDA compatible)
- Dependencies listed in `requirements.txt`

Install dependencies with:
pip install -r requirements.txt
textNote: If using a GPU, ensure PyTorch is installed with CUDA support. You can check installation instructions at https://pytorch.org/get-started/locally/.

## Dataset
Download the dataset from: https://www.kaggle.com/datasets/rishiikumarsingh/duality-offroad-segmentation/data

Unzip the downloaded file. The dataset should have the following structure:
data/
├── train/
│   ├── Color_Images/
│   └── Segmentation/
├── val/
│   ├── Color_Images/
│   └── Segmentation/
└── test/
├── Color_Images/
└── Segmentation/
textPlace this `data/` folder in your working directory or specify the path using `--data_dir`.

## Step-by-Step Instructions

### 1. Training the Model
Run the training script:
python train.py --data_dir path/to/data --output_dir path/to/outputs
text- `--data_dir`: Path to the dataset root directory (containing train/val/test).
- `--output_dir`: Directory to save the best model (`best_model.pth`) and plots (default: `outputs`).

The script will:
- Train the model for 10 epochs.
- Print training and validation metrics (Loss, IoU, Pixel Accuracy) for each epoch.
- Save the model with the best validation IoU.
- Generate and save plots: `loss_plot.png`, `iou_plot.png`, `acc_plot.png`.

Expected runtime: Approximately 1-2 hours on a GPU like NVIDIA T4 (depending on hardware).

### 2. Testing the Model
After training, run the test script:
python test.py --data_dir path/to/data --model_path path/to/outputs/best_model.pth --output_dir path/to/outputs
text- `--data_dir`: Same as above.
- `--model_path`: Path to the saved model file.
- `--output_dir`: Directory to save `test_metrics.txt` (default: `outputs`).

The script will:
- Evaluate the model on the test set using Test-Time Augmentation (TTA: horizontal flip).
- Compute and print mean Test Loss, IoU, and Pixel Accuracy.
- Save the metrics to `test_metrics.txt`.

## Reproducing Final Results
1. Download and prepare the dataset as described.
2. Install dependencies.
3. Run `train.py` to train and save the model.
4. Run `test.py` using the saved model.
5. Compare the output metrics with those from the original notebook (expect slight variations due to randomness in training).

To get identical results, you may need to set random seeds (not implemented in the current scripts), but the metrics should be similar.

## Expected Outputs and Interpretation
- **During Training**:
  - Console output shows per-epoch metrics.
  - Example: `Epoch 1/10 | Train Loss: X.XXXX | Train IoU: X.XXXX | ...`
  - Plots visualize learning curves.

- **During Testing**:
  - Console output: `Mean Test Loss: X.XXXX`, `Mean Test IoU: X.XXXX`, `Mean Test Pixel Accuracy: X.XXXX`
  - `test_metrics.txt`: Text file with the same metrics.

- **Interpretation**:
  - **Loss**: Lower is better (combined CE + Dice + Lovasz).
  - **IoU (Intersection over Union)**: Higher is better (0-1 scale); measures overlap between predicted and ground truth segments. Mean IoU across classes, ignoring NaNs for classes not present.
  - **Pixel Accuracy**: Higher is better (0-1 scale); fraction of pixels correctly classified.

If you encounter issues (e.g., out-of-memory), reduce batch_size in the scripts.

For questions, refer to the original Kaggle notebook or dataset page.
