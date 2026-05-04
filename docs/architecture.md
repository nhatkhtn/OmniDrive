# Test-Time Architecture ([mask_eva_lane_det_vlm.py#L1-L1](https://github.com/nhatkhtn/OmniDrive/blob/1c96135300de667073716efa283478c0aaab641c/projects/configs/OmniDrive/mask_eva_lane_det_vlm.py#L1-L1))

This document is limited to the test-time path used by the config and [dist_test.sh#L1-L16](https://github.com/nhatkhtn/OmniDrive/blob/1c96135300de667073716efa283478c0aaab641c/tools/dist_test.sh#L1-L16).

## 1. Main model components and where they live

- Petr3D detector (overall orchestrator): [petr3d.py#L40-L40](https://github.com/nhatkhtn/OmniDrive/blob/1c96135300de667073716efa283478c0aaab641c/projects/mmdet3d_plugin/models/detectors/petr3d.py#L40-L40)
  - Builds the backbone, heads, and LLM, runs `forward_test`, and concatenates visual tokens for the LLM.
- EVAViT image backbone: [eva_vit.py#L862-L862](https://github.com/nhatkhtn/OmniDrive/blob/1c96135300de667073716efa283478c0aaab641c/projects/mmdet3d_plugin/models/backbones/eva_vit.py#L862-L862)
  - Produces multiview image features used by both heads.
- StreamPETRHead (3D detection head): [streampetr_head.py#L38-L38](https://github.com/nhatkhtn/OmniDrive/blob/1c96135300de667073716efa283478c0aaab641c/projects/mmdet3d_plugin/models/dense_heads/streampetr_head.py#L38-L38)
  - Produces detection outputs and the VLM token block (`vlm_memory`) used by the LLM.
- PETRHeadM (map/lane head): [petr_head_map.py#L40-L40](https://github.com/nhatkhtn/OmniDrive/blob/1c96135300de667073716efa283478c0aaab641c/projects/mmdet3d_plugin/models/dense_heads/petr_head_map.py#L40-L40)
  - Produces lane outputs and the VLM token block (`vlm_memory`) used by the LLM.
- LlavaLlamaForCausalLM (LLM wrapper): [llava_llama.py#L48-L48](https://github.com/nhatkhtn/OmniDrive/blob/1c96135300de667073716efa283478c0aaab641c/projects/mmdet3d_plugin/models/dense_heads/llava_llama.py#L48-L48)
  - Replaces the `<image>` sentinel with visual tokens and runs generation.
- LLM loader helper: [misc.py#L232-L232](https://github.com/nhatkhtn/OmniDrive/blob/1c96135300de667073716efa283478c0aaab641c/projects/mmdet3d_plugin/models/utils/misc.py#L232-L232)
  - Loads the base LLM weights from `llm_path` during model init.

## 2. Checkpoint files used at test time

Used in this test run:

- ckpts/OmniDrive/iter_10548.pth
  - Loaded by the test runner via `load_checkpoint` for the full model. This is the main test-time checkpoint that updates the backbone, heads, and LLM weights as present in the file. See [test.py#L212-L212](https://github.com/nhatkhtn/OmniDrive/blob/1c96135300de667073716efa283478c0aaab641c/mmdetection3d/tools/test.py#L212-L212).
- ckpts/pretrain_qformer/
  - Used to initialize the tokenizer and the LlavaLlamaForCausalLM instance before checkpoint loading. See [mask_eva_lane_det_vlm.py#L25-L41](https://github.com/nhatkhtn/OmniDrive/blob/1c96135300de667073716efa283478c0aaab641c/projects/configs/OmniDrive/mask_eva_lane_det_vlm.py#L25-L41) and [petr3d.py#L139-L139](https://github.com/nhatkhtn/OmniDrive/blob/1c96135300de667073716efa283478c0aaab641c/projects/mmdet3d_plugin/models/detectors/petr3d.py#L139-L139).

Present in download instructions but not used by this test command:

- ckpts/eva02_petr_proj.pth
  - Listed as `load_from` in the config (training init). This is not used by [dist_test.sh#L1-L16](https://github.com/nhatkhtn/OmniDrive/blob/1c96135300de667073716efa283478c0aaab641c/tools/dist_test.sh#L1-L16), which loads the checkpoint passed on the command line instead. See [mask_eva_lane_det_vlm.py#L295-L295](https://github.com/nhatkhtn/OmniDrive/blob/1c96135300de667073716efa283478c0aaab641c/projects/configs/OmniDrive/mask_eva_lane_det_vlm.py#L295-L295).

## 3. Dataset files used (test-time pipeline)

Main annotation file (always used in test):

- data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl
  - Structure: a dict with key `infos`, where `infos` is a list of per-sample dicts. See [nuscenes_dataset.py#L103-L103](https://github.com/nhatkhtn/OmniDrive/blob/1c96135300de667073716efa283478c0aaab641c/projects/mmdet3d_plugin/datasets/nuscenes_dataset.py#L103-L103).
  - Keys used in test-time input construction:
    - `token` -> `sample_idx` (used for output file names and meta).
    - `location` -> location prefix in the prompt.
    - `timestamp` (microseconds) -> `timestamp` input after scaling.
    - `gt_planning_command` -> `command` input.
    - `can_bus` -> `can_bus` input (length 13 vector).
    - `cams` -> dict of 6 cameras with per-camera fields:
      - `data_path` -> image filenames.
      - `cam_intrinsic`, `sensor2ego_rotation`, `sensor2ego_translation`, `timestamp` -> intrinsics/extrinsics, lidar2img, and `img_timestamp`.
    - `ego2global_rotation`, `ego2global_translation` -> `ego_pose` / `ego_pose_inv`.

Image files used by the pipeline:

- Paths stored in each `cams[cam_name]['data_path']`, loaded by LoadMultiViewImageFromFiles.

Configured but not read when `load_type=["planning"]`:

- data/nuscenes/conv/val/<token>.json
- data/nuscenes/vqa/val/<token>.json
- data/nuscenes/eval_cf/<token>.pkl

These are only used when `LoadAnnoatationVQATest.load_type` includes `conv` or `counter`. In this test, `load_type=["planning"]` (see [mask_eva_lane_det_vlm.py#L207-L214](https://github.com/nhatkhtn/OmniDrive/blob/1c96135300de667073716efa283478c0aaab641c/projects/configs/OmniDrive/mask_eva_lane_det_vlm.py#L207-L214)), so only the planning question is generated and no external QA files are read.

## 4. Main forward function and test-time inputs (exact shapes)

Entry point for test-time inference:

- Petr3D.forward_test in [petr3d.py#L394-L394](https://github.com/nhatkhtn/OmniDrive/blob/1c96135300de667073716efa283478c0aaab641c/projects/mmdet3d_plugin/models/detectors/petr3d.py#L394-L394)
  - Signature: `forward_test(self, img_metas, rescale, **data)`
  - It normalizes the nested test-time data containers, then calls `simple_test(...)`.

Inputs after `forward_test` normalization (B=1, N=6):

- `img`: float32, shape [1, 6, 3, 640, 640]
  - From `cams[*]['data_path']` -> LoadMultiViewImageFromFiles -> ResizeCropFlipRotImage -> ResizeMultiview3D(img_scale=(640,640)) -> PadMultiViewImage -> PETRFormatBundle3D.
- `lidar2img`: float32, shape [1, 6, 4, 4]
  - Built from per-camera intrinsics/extrinsics and updated in ResizeCropFlipRotImage ([transform_3d.py#L312-L312](https://github.com/nhatkhtn/OmniDrive/blob/1c96135300de667073716efa283478c0aaab641c/projects/mmdet3d_plugin/datasets/pipelines/transform_3d.py#L312-L312)).
- `intrinsics`: float32, shape [1, 6, 4, 4]
  - From `cams[*]['cam_intrinsic']`, padded to 4x4, then updated by the image augmentations.
- `extrinsics`: float32, shape [1, 6, 4, 4]
  - From `cams[*]['sensor2ego_rotation/translation']`, converted to 4x4.
- `timestamp`: float64, shape [1]
  - From `info['timestamp'] / 1e6`.
- `img_timestamp`: float64, shape [1, 6]
  - Per-camera timestamps (seconds).
- `ego_pose`: float32, shape [1, 4, 4]
  - From `ego2global_rotation/translation`.
- `ego_pose_inv`: float32, shape [1, 4, 4]
  - Inverse of `ego_pose`.
- `command`: float32, shape [1]
  - From `gt_planning_command`.
- `can_bus`: float32, shape [1, 13]
  - From `can_bus` in the annotation file.
- `input_ids`: list of length B, each element is a list of Q 1D int64 tensors
  - With `load_type=["planning"]`, Q=1.
  - Each tensor has variable length L (tokenized prompt length).
- `img_metas`: list of length 1 with meta keys (including `sample_idx`, `vlm_labels`, and image shapes).

## 5. LLM input construction (full input_ids length)

Text prompt source (test-only):

- LoadAnnoatationVQATest builds one question: "Please provide the planning trajectory for the ego car without reasons." ([transform_3d.py#L765-L765](https://github.com/nhatkhtn/OmniDrive/blob/1c96135300de667073716efa283478c0aaab641c/projects/mmdet3d_plugin/datasets/pipelines/transform_3d.py#L765-L765)).
- It prepends `<image>\nYou are driving in {location}. ` and then tokenizes with the Vicuna-v1 template ([transform_3d.py#L836-L836](https://github.com/nhatkhtn/OmniDrive/blob/1c96135300de667073716efa283478c0aaab641c/projects/mmdet3d_plugin/datasets/pipelines/transform_3d.py#L836-L836), [conversation.py#L252-L252](https://github.com/nhatkhtn/OmniDrive/blob/1c96135300de667073716efa283478c0aaab641c/projects/mmdet3d_plugin/datasets/utils/conversation.py#L252-L252), and [data_utils.py#L167-L167](https://github.com/nhatkhtn/OmniDrive/blob/1c96135300de667073716efa283478c0aaab641c/projects/mmdet3d_plugin/datasets/utils/data_utils.py#L167-L167)).

Visual token source:

- StreamPETRHead returns `vlm_memory` from the first `num_extra` decoder slots, then appends a single `can_bus` embedding token ([streampetr_head.py#L558-L588](https://github.com/nhatkhtn/OmniDrive/blob/1c96135300de667073716efa283478c0aaab641c/projects/mmdet3d_plugin/models/dense_heads/streampetr_head.py#L558-L588)).
- PETRHeadM returns `vlm_memory` from its first `num_extra` decoder slots ([petr_head_map.py#L429-L429](https://github.com/nhatkhtn/OmniDrive/blob/1c96135300de667073716efa283478c0aaab641c/projects/mmdet3d_plugin/models/dense_heads/petr_head_map.py#L429-L429)).
- Petr3D concatenates both into `vision_embeded` and passes it to the LLM ([petr3d.py#L436-L436](https://github.com/nhatkhtn/OmniDrive/blob/1c96135300de667073716efa283478c0aaab641c/projects/mmdet3d_plugin/models/detectors/petr3d.py#L436-L436)).

Exact length formula (per prompt):

- Let $L_{raw}$ be the token length of the Vicuna-v1 text sequence, which includes exactly one `<image>` sentinel.
- Let $T_{vision}$ be the number of visual tokens, which equals `vision_embeded.shape[1]`.
- The LLM input length is:

$$L_{final} = (L_{raw} - 1) + T_{vision}$$

In this config:

- `num_extra` in StreamPETRHead = 256
- `num_extra` in PETRHeadM = 256
- StreamPETRHead adds one extra token for `can_bus`

So $T_{vision} = 256 + 1 + 256 = 513$.

## 6. Test-time outputs

The model returns a list with one dict per sample (B=1) in Petr3D.simple_test ([petr3d.py#L458-L468](https://github.com/nhatkhtn/OmniDrive/blob/1c96135300de667073716efa283478c0aaab641c/projects/mmdet3d_plugin/models/detectors/petr3d.py#L458-L468)):

- `pts_bbox`: 3D detection results from StreamPETRHead (boxes, scores, labels).
- `lane_results`: lane centerline results from PETRHeadM.
- `text_out`: list of QA dicts `{Q, A}` produced by the LLM.

Side effect:

- If the sample output file does not already exist, the QA list is also written to `save_path` (default `./results_planning_only/`) using the `sample_idx` as the filename.
