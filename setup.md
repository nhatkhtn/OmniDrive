Note:
- The whole repo only contains Omni-Q code and pre-trained weights, but not Omni-L.
- The ortools requirement in OmniLane-V2 needs to be modified to ">=" instead of "==", because the specified version does not exist.

curl -Z --parallel-max 2 -O -C - "https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval[01-10]_blobs.tgz"

uv run hf download exiawsh/pretrain_qformer --local-dir ./ckpts/pretrain_qformer

uv run hf download exiawsh/OmniDrive --local-dir ./ckpts/OmniDrive