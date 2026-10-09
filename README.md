## Code Availability

We are currently refining and updating the codebase to incorporate our latest research improvements. The official implementation will be released soon. Thank you for your interest and patience.

# LCMCA-Net

LCMCA-Net incorporates a contour-aware module with large-kernel convolutional attention and a deformable corner alignment module.

Our paper is available at [IEEE Xplore](https://ieeexplore.ieee.org/document/11698713).

Installation, training, and validation instructions will be provided upon the official code release.

## 1.Environment Install

#codna environment
conda create -n od python=3.8
conda activate od

#cuda=11.8,pytorch=2.0.0
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

#mmdetection
cd mmpycocotools
python setup.py develop
cd ../
pip install -e .

## 2.Train
### 2.1 DIOR dataset

nohup bash tools/dist_train_dior.sh \
    configs/pycenternet/lcmca_tiny_adamw_6x.py 8 \
    > train_dior.log 2>&1 & disown

### 2.2 NWPU VAR-10 dataset

nohup bash tools/dist_train_nwpu.sh \
    configs/pycenternet/nwpu_lcmca_tiny_adamw_6x.py 8 \
    > train_nwpu.log 2>&1 & disown

## 3.Validate

bash ./tools/dist_test_x.sh \
   work_dirs/dior/xx/xx.py\
   work_dirs/dior/xx/latest.pth \
    8 \
    --eval bbox \
    --eval-options "classwise=True"

