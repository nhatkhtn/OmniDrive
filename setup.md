Note:
- The whole repo only contains Omni-Q code and pre-trained weights, but not Omni-L.
- The ortools requirement in OmniLane-V2 needs to be modified to ">=9.2,<9.4" instead of "==9.2.9972", because the specified version does not exist.
- The networkx requirement in mmdetection3d needs to be modified to ">=2.5,<2.6" instead of ">=2.2,<2.3", because this repo itself requires networkx>=2.5.

```bash
curl -Z --parallel-max 2 -O -C - "https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval[01-10]_blobs.tgz"
```

The file below seems to be not used in testing.
```bash
curl -L -O https://github.com/NVlabs/OmniDrive/releases/download/v1.0/eva02_petr_proj.pth
```

```bash
curl -L -O https://github.com/NVlabs/OmniDrive/releases/download/v1.0/data_nusc.zip
```

```bash
curl -L -O  https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval_meta.tgz
curl -L -O https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/nuScenes-map-expansion-v1.3.zip
```

```bash
uv run hf download exiawsh/pretrain_qformer --local-dir ./ckpts/pretrain_qformer
```

```bash
uv run hf download exiawsh/OmniDrive --local-dir ./ckpts/OmniDrive
```

```bash
PORT=29501 CUDA_VISIBLE_DEVICES=0 uv run tools/dist_test.sh projects/configs/OmniDrive/mask_eva_lane_det_vlm.py ckpts/OmniDrive/iter_10548.pth 1 --format-only
```

```bash
PORT=29501 uv run tools/dist_test.sh projects/configs/OmniDrive/mask_eva_lane_det_vlm.py ckpts/OmniDrive/iter_10548.pth 2 --format-only
```

```bash
uv run eval_planning.py --pred_path ../results_planning_only/ --anno_path nuscenes2d_ego_temporal_infos_val.pkl
```

# Architecture

Here is the corrected test-time summary, with the geometry part made explicit and the input shapes stated as the code uses them.

**Architecture**
OmniDrive is built as a single `Petr3D` model in petr3d.py. The config in mask_eva_lane_det_vlm.py wires three main parts:
- `img_backbone = EVAViT`
- `pts_bbox_head = StreamPETRHead` from projects/mmdet3d_plugin/models/dense_heads/streampetr_head.py
- `map_head = PETRHeadM` from projects/mmdet3d_plugin/models/dense_heads/petr_head_map.py
- `lm_head = LlavaLlamaForCausalLM` from deploy/llm_lib/llava_llama.py

**Test-time inputs and routes**
- `img` has shape `[B, N, 3, H, W]`, where `B` is batch size and `N` is number of cameras. It goes through `extract_img_feat` and then the EVAViT backbone, before being passed into `pts_bbox_head` and `map_head` via `data['img_feats']` in petr3d.py and petr3d.py.
- `lidar2img` is treated as `[B, N, 4, 4]`. In `position_embeding`, the code explicitly calls `.inverse()` and reshapes it to `[B, LEN, D, 4, 4]` before lifting 2D locations into 3D-aware coordinates in petr3d.py.
- `intrinsics` is a per-camera calibration matrix. The code reads the focal terms from `intrinsics[..., 0, 0]` and `intrinsics[..., 1, 1]` in the same `position_embeding` function, so the only shape guarantee the code makes is that those entries exist.
- `timestamp` is used as a per-sample scalar or 1D tensor in the heads’ memory logic. It is consumed inside the temporal alignment and memory update code in both heads, not by the LM.
- `ego_pose` and `ego_pose_inv` are homogeneous pose matrices used by the temporal memory code in the heads. The heads multiply and transform their memory state with these matrices.
- `command` and `can_bus` are combined inside `StreamPETRHead` into a 14-D vehicle-state vector. That vector is then folded into the temporal memory branch.
- `input_ids` is the text prompt input. At test time, `Petr3D` passes it to `self.lm_head.generate(...)` together with the visual token sequence.
- `sample_idx` and `vlm_labels` are only used for output bookkeeping and saving results, not as model inputs at test time.

**How the inputs flow**
The key geometry step is `prepare_location` plus `position_embeding` in petr3d.py. In plain terms:
- `img` gives feature maps through EVAViT.
- `prepare_location` builds a 2D grid of feature-map locations.
- `position_embeding` uses `lidar2img` and `intrinsics` to turn those image-grid locations into 3D-aware positional embeddings.
- Those embeddings go into both heads.
- `pts_bbox_head` produces `det_query` and `map_head` produces `map_query`.
- `Petr3D` concatenates them into `vision_embeded`.
- `vision_embeded` and `input_ids` then go into the LLaVA-style LM wrapper, which receives them as `images=vision_embeded` in petr3d.py.

So the “already-computed query embeddings” are the head outputs, not raw image patches. They are token sequences distilled by the detection and map heads from the camera backbone features.

**Weights**
At test time, the actual checkpoint loaded by test.py is iter_10548.pth, because test.py calls `load_checkpoint(model, args.checkpoint, ...)` and does not load `load_from` during testing. That file contains the weights for:
- `img_backbone`
- `pts_bbox_head`
- `map_head`
- `lm_head`

The config also sets:
- `llm_path = 'ckpts/pretrain_qformer/'` in mask_eva_lane_det_vlm.py, which is used to construct the tokenizer and LM wrapper
- `load_from = 'ckpts/eva02_petr_proj.pth'` in mask_eva_lane_det_vlm.py, but this is a training/init setting, not what dist_test.sh actually loads

If you want the shortest possible one-line version:  
`img` goes to EVAViT, calibration and pose tensors go to the geometry/memory code in the two heads, the heads produce query tokens, and those tokens plus `input_ids` go to the LLaVA-style LM.