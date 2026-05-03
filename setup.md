Note:
- The whole repo only contains Omni-Q code and pre-trained weights, but not Omni-L.
- The ortools requirement in OmniLane-V2 needs to be modified to ">=9.2,<9.4" instead of "==9.2.9972", because the specified version does not exist.
- The networkx requirement in mmdetection3d needs to be modified to ">=2.5,<2.6" instead of ">=2.2,<2.3", because this repo itself requires networkx>=2.5.

curl -Z --parallel-max 2 -O -C - "https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval[01-10]_blobs.tgz"

curl -L -O https://github.com/NVlabs/OmniDrive/releases/download/v1.0/eva02_petr_proj.pth

curl -L -O https://github.com/NVlabs/OmniDrive/releases/download/v1.0/data_nusc.zip

uv run hf download exiawsh/pretrain_qformer --local-dir ./ckpts/pretrain_qformer

uv run hf download exiawsh/OmniDrive --local-dir ./ckpts/OmniDrive

PORT=29501 CUDA_VISIBLE_DEVICES=0 uv run tools/dist_test.sh projects/configs/OmniDrive/mask_eva_lane_det_vlm.py ckpts/OmniDrive/iter_10548.pth 1 --format-only
PORT=29501 uv run tools/dist_test.sh projects/configs/OmniDrive/mask_eva_lane_det_vlm.py ckpts/OmniDrive/iter_10548.pth 2 --format-only