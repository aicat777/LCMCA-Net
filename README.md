# LCMCA-Net

LCMCA-Net incorporates a contour-aware module with large-kernel convolutional attention and a deformable corner alignment module.

Our paper is available at [IEEE Xplore](https://ieeexplore.ieee.org/document/11698713).

## Code Availability

We are currently refining and updating the codebase to incorporate our latest research improvements. The official implementation will be released soon. Thank you for your interest and patience.

The following instructions are provided for reference and will be executable once the official codebase is released.

## 1. Environment Setup

Create and activate the Conda environment:

```bash
# Conda environment
conda create -n od python=3.8
conda activate od
```

Install PyTorch with CUDA 11.8:

```bash
# CUDA 11.8, PyTorch 2.0.0
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
```

Install MMDetection and the required dependencies:

```bash
# MMDetection
cd mmpycocotools
python setup.py develop
cd ../
pip install -e .
```

## 2. Training

### 2.1 DIOR Dataset

To train LCMCA-Net on the DIOR dataset using 8 GPUs, run:

```bash
nohup bash tools/dist_train_dior.sh \
    configs/pycenternet/lcmca_tiny_adamw_6x.py 8 \
    > train_dior.log 2>&1 & disown
```

### 2.2 NWPU VHR-10 Dataset

To train LCMCA-Net on the NWPU VHR-10 dataset using 8 GPUs, run:

```bash
nohup bash tools/dist_train_nwpu.sh \
    configs/pycenternet/nwpu_lcmca_tiny_adamw_6x.py 8 \
    > train_nwpu.log 2>&1 & disown
```

## 3. Evaluation

To evaluate the trained model, run:

```bash
bash ./tools/dist_test_x.sh \
    work_dirs/dior/xx/xx.py \
    work_dirs/dior/xx/latest.pth \
    8 \
    --eval bbox \
    --eval-options "classwise=True"
```

Replace the configuration and checkpoint paths with the corresponding files for your experiment.
