"""
Smoke-test script for OmniDrive (NOT part of upstream OmniDrive; added at the
submodule root per task rules -- never edits any upstream-shipped file).

Runs ONE forward pass of the OmniDrive-Agent (Vicuna-7B + StreamPETR-style
3D perception) on the pinned DriveBench MCQ sample at
`data/test_sample.json` (path relative to the outer repo root).

Documented shortcuts (this is a smoke test, not an evaluation run):

  1. Real 6-camera images ARE used (matched to the pinned CAM_BACK frame by
     nuScenes timestamp prefix across the other 5 CAM_* directories), but
     per-camera calibration (intrinsics/extrinsics) is a DUMMY: identity
     extrinsics (lidar2cam = I) and one shared, plausible nuScenes-scale
     pinhole intrinsic matrix for all 6 cams. We do not have the released
     `nuscenes2d_ego_temporal_infos_*.pkl` / raw nuScenes v1.0 tables needed
     to compute the real per-camera calibration for this frame, and
     building them was out of scope for a smoke test. can_bus / ego_pose /
     command are dummy zeros/identity too.
  2. Instead of the full `CustomNuScenesDataset` + on-disk VQA json files,
     we register one extra pipeline stage (`LoadSingleQuestionVQA`) that
     tokenizes the ONE question from data/test_sample.json using the exact
     same helpers (`preprocess`, vicuna_v1 conversation template) upstream
     uses in `LoadAnnoatationVQATest`. Every other pipeline stage is the
     real upstream implementation, composed exactly like
     `projects/configs/OmniDrive/mask_eva_lane_det_vlm.py`'s test_pipeline.
  3. Runs on GPU (B200, sm_100) with torch 2.7.0+cu128 and flash_attn 2.8.0
     -- the working env the user pins in `pyproject.toml`. Model + inputs
     go to cuda:0; the trained fp16 checkpoint loads and runs in fp16.
     `uv sync --project OmniDrive --frozen` installs the wrong mmcv-full
     (openmmlab's cu117 wheel has no torch-2.7 build); after syncing, force
     -install the vendored one from wheels/, plus peft/sentencepiece which
     aren't in the lock either -- see wheels/README.md for exact commands
     and why this isn't wired into uv.lock automatically.
  4. `generate()`'s `max_new_tokens=320` (hardcoded in upstream
     `Petr3D.simple_test_pts`) is capped down via a runtime monkeypatch of
     the bound `model.lm_head.generate` method (not an edit to the
     upstream file) so the smoke test finishes in seconds instead of
     generating a long paragraph we don't need.
  5. Model weights are loaded from the plain `ckpts/` layout that
     `OmniDrive/setup.md` documents, found already downloaded at
     `/blue/thai/hoangx/repos/OmniDrive/ckpts/` on this cluster (the
     `exiawsh/OmniDrive` and `exiawsh/pretrain_qformer` HF hub cache
     entries under `~/.cache/huggingface/hub` turned out to be empty
     stubs -- refs only, no snapshots/blobs -- so this plain-layout copy is
     the one actually usable).
"""
import argparse
import glob
import json
import os
import sys

OMNIDRIVE_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(OMNIDRIVE_ROOT)
if OMNIDRIVE_ROOT not in sys.path:
    sys.path.insert(0, OMNIDRIVE_ROOT)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

CKPT_ROOT = "/blue/thai/hoangx/repos/OmniDrive/ckpts"
LLM_PATH = os.path.join(CKPT_ROOT, "pretrain_qformer")
CHECKPOINT_PATH = os.path.join(CKPT_ROOT, "OmniDrive", "iter_10548.pth")
SAVE_PATH = "/tmp/omnidrive_smoketest_results/"
MAX_NEW_TOKENS = 32  # capped for the smoke test; upstream default is 320

CAMS = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
        "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

CLASS_NAMES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

IDA_AUG_CONF = {
    "resize_lim": (0.37, 0.45),
    "final_dim": (320, 640),
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 900,
    "W": 1600,
    "rand_flip": False,
}

IMG_NORM_CFG = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

COLLECT_KEYS = ['lidar2img', 'intrinsics', 'extrinsics', 'timestamp',
                'img_timestamp', 'ego_pose', 'ego_pose_inv', 'command', 'can_bus']


def find_cam_images(repo_root, back_image_path):
    """Given the CAM_BACK path from test_sample.json, find the other 5
    camera images for the SAME frame by matching the nuScenes timestamp
    prefix (all 6 cameras fire within a few ms of each other and share the
    first 9 digits of the microsecond timestamp for this scene)."""
    back_abs = os.path.join(repo_root, back_image_path)
    basename = os.path.basename(back_abs)
    scene_prefix, _, rest = basename.partition("__CAM_BACK__")
    ts_prefix = rest[:9]
    samples_root = os.path.join(repo_root, "DriveBench", "data", "nuscenes", "samples")

    found = {}
    for cam in CAMS:
        pattern = os.path.join(samples_root, cam, f"{scene_prefix}__{cam}__{ts_prefix}*.jpg")
        matches = sorted(glob.glob(pattern))
        assert matches, f"no image found for {cam} matching {pattern}"
        found[cam] = matches[0]
    return found


def build_raw_sample(question_json_path, repo_root):
    with open(question_json_path) as f:
        sample = json.load(f)

    cam_images = find_cam_images(repo_root, sample["image_path"]["CAM_BACK"])
    img_filenames = [cam_images[c] for c in CAMS]

    import numpy as np
    from mmdet3d.core.bbox import get_box_type

    # --- Documented dummy calibration (shortcut #1 above) ---
    intrinsic = np.array([
        [1266.4, 0.0, 816.3, 0.0],
        [0.0, 1266.4, 491.5, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)
    extrinsic = np.eye(4, dtype=np.float32)  # lidar2cam = identity (dummy)

    intrinsics = [intrinsic.copy() for _ in CAMS]
    extrinsics = [extrinsic.copy() for _ in CAMS]
    lidar2img = [intrinsics[i] @ extrinsics[i] for i in range(len(CAMS))]

    box_type_3d, box_mode_3d = get_box_type('LiDAR')

    raw = dict(
        img_filename=img_filenames,
        filename=img_filenames,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        lidar2img=lidar2img,
        img_timestamp=[0.0 for _ in CAMS],
        timestamp=0.0,
        ego_pose=np.eye(4, dtype=np.float32),
        ego_pose_inv=np.eye(4, dtype=np.float32),
        can_bus=np.zeros(13, dtype=np.float32),
        command=0.0,
        location="boston",  # n008-* scenes are boston-seaport; only used for the text prompt
        sample_idx=sample["frame_token"],
        scene_token=sample["scene_token"],
        prev_idx='',
        next_idx='',
        frame_idx=0,
        box_type_3d=box_type_3d,
        box_mode_3d=box_mode_3d,
        question=sample["question"],
        bbox3d_fields=[],
    )
    return raw, sample


def build_test_pipeline():
    return [
        dict(type='LoadMultiViewImageFromFiles', to_float32=True),
        dict(type='ResizeCropFlipRotImage', data_aug_conf=IDA_AUG_CONF, training=False),
        dict(type='ResizeMultiview3D', img_scale=(640, 640), keep_ratio=False, multiscale_mode='value'),
        dict(type='LoadSingleQuestionVQA', tokenizer=LLM_PATH, max_length=2048),
        dict(type='NormalizeMultiviewImage', **IMG_NORM_CFG),
        dict(type='PadMultiViewImage', size_divisor=32),
        dict(
            type='MultiScaleFlipAug3D',
            img_scale=(1333, 800),
            pts_scale_ratio=1,
            flip=False,
            transforms=[
                dict(
                    type='PETRFormatBundle3D',
                    collect_keys=COLLECT_KEYS,
                    class_names=CLASS_NAMES,
                    with_label=False),
                dict(type='Collect3D', keys=['input_ids', 'img'] + COLLECT_KEYS,
                     meta_keys=('sample_idx', 'vlm_labels', 'filename', 'ori_shape', 'img_shape',
                                'pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d',
                                'img_norm_cfg', 'scene_token'))
            ])
    ]


def register_single_question_pipeline():
    from mmdet.datasets.builder import PIPELINES
    from transformers import AutoTokenizer
    from projects.mmdet3d_plugin.datasets.utils.data_utils import preprocess
    from projects.mmdet3d_plugin.datasets.utils.constants import DEFAULT_IMAGE_TOKEN

    if 'LoadSingleQuestionVQA' in PIPELINES:
        return

    @PIPELINES.register_module()
    class LoadSingleQuestionVQA:
        """New (non-upstream) pipeline stage: tokenizes the ONE question
        carried on `results['question']`, replacing LoadAnnoatationVQATest's
        on-disk-json based question loading for this smoke test. Uses the
        exact same tokenization helper + vicuna_v1 conversation template as
        upstream so the resulting input_ids match what the model expects."""

        def __init__(self, tokenizer, max_length=2048):
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer, model_max_length=max_length, padding_side="right", use_fast=False)
            self.tokenizer.pad_token = self.tokenizer.unk_token

        def __call__(self, results):
            question = results['question']
            prompt = f"You are driving in {results['location']}. "
            sources = [[
                {"from": "human", "value": DEFAULT_IMAGE_TOKEN + "\n" + prompt + question},
                {"from": "gpt", "value": ""},
            ]]
            vlm_labels = [question]
            converted = preprocess(sources, self.tokenizer, True, False)
            results['input_ids'] = converted['input_ids']
            results['vlm_labels'] = vlm_labels
            return results

        def __repr__(self):
            return self.__class__.__name__


def check_env():
    """Fail fast with a clear message if the env wasn't set up via
    wheels/setup_env.sh (e.g. someone ran a bare `uv sync`, which silently
    prunes mmcv-full/peft/sentencepiece/the cu11 shim back out since none of
    them are declared in pyproject.toml/uv.lock -- see wheels/README.md)."""
    import importlib
    missing = []
    for mod in ("peft", "sentencepiece"):
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        sys.exit(
            f"OmniDrive env is missing {missing} (not declared in "
            "pyproject.toml/uv.lock -- see wheels/README.md). Run "
            "wheels/setup_env.sh instead of a bare `uv sync`."
        )
    try:
        import mmcv.ops  # noqa: F401  fails if the wrong (cu117/torch1.13) wheel is installed
    except (ImportError, OSError) as e:
        sys.exit(
            "mmcv-full looks like the wrong build (probably openmmlab's "
            f"cu117/torch1.13 stub, not the vendored torch-2.7/cu128 one): {e}\n"
            "Run wheels/setup_env.sh instead of a bare `uv sync`."
        )


def main():
    check_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default=os.path.join(REPO_ROOT, "data", "test_sample.json"))
    parser.add_argument("--config", default=os.path.join(
        OMNIDRIVE_ROOT, "projects", "configs", "OmniDrive", "mask_eva_lane_det_vlm.py"))
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    args = parser.parse_args()

    import torch
    import mmcv
    from mmcv import Config
    from mmcv.runner import load_checkpoint
    from mmcv.parallel import collate, MMDataParallel
    # NOTE: mmdet3d keeps its own separate PIPELINES registry (not linked to
    # mmdet's), so LoadMultiViewImageFromFiles/Collect3D/MultiScaleFlipAug3D
    # are only visible via mmdet3d's own Compose (which falls back to
    # mmdet's registry for anything it doesn't find, e.g. our custom stage).
    from mmdet3d.datasets.pipelines import Compose
    import importlib
    import mmdet3d.datasets  # noqa: F401  registers LoadMultiViewImageFromFiles/Collect3D/MultiScaleFlipAug3D
    importlib.import_module('projects.mmdet3d_plugin')
    register_single_question_pipeline()
    from mmdet3d.models import build_model

    assert torch.cuda.is_available(), "This script runs OmniDrive on GPU; no CUDA device visible."

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.tokenizer = LLM_PATH
    cfg.model.lm_head = LLM_PATH
    cfg.model.save_path = SAVE_PATH

    print(f"[run_sample_omnidrive] building model from {args.config} ...", flush=True)
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    print(f"[run_sample_omnidrive] loading checkpoint {args.checkpoint} ...", flush=True)
    load_checkpoint(model, args.checkpoint, map_location='cpu', strict=False)
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    # torch 2.7 changed `_get_stream(int)` -> `_get_stream(torch.device)`.
    # mmcv 1.7 still passes ints, so wrap it. Patch BOTH the source in torch AND
    # mmcv's local rebinding (mmcv does `from torch.nn.parallel._functions import _get_stream`).
    import torch.nn.parallel._functions as _tnpf
    import mmcv.parallel._functions as _mpf
    _orig_get_stream = _tnpf._get_stream
    def _int_safe_get_stream(dev):
        if isinstance(dev, int):
            dev = torch.device('cuda', dev)
        return _orig_get_stream(dev)
    _tnpf._get_stream = _int_safe_get_stream
    _mpf._get_stream = _int_safe_get_stream

    # Cap max_new_tokens (upstream hardcodes 320 in Petr3D.simple_test_pts).
    # Monkeypatch the bound method rather than editing the upstream file.
    _orig_generate = model.module.lm_head.generate

    def _capped_generate(*args, **kwargs):
        kwargs['max_new_tokens'] = min(kwargs.get('max_new_tokens', MAX_NEW_TOKENS), MAX_NEW_TOKENS)
        return _orig_generate(*args, **kwargs)

    model.module.lm_head.generate = _capped_generate

    raw_sample, sample_json = build_raw_sample(args.sample, REPO_ROOT)
    pipeline = Compose(build_test_pipeline())
    data = pipeline(raw_sample)
    data = collate([data], samples_per_gpu=1)

    print("[run_sample_omnidrive] question:", sample_json["question"], flush=True)
    print("[run_sample_omnidrive] ground-truth answer (not required to match):", sample_json["answer"], flush=True)

    # MMDataParallel handles DataContainer scatter to cuda:0 and unwraps.
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)

    text_out = result[0]['text_out']
    print("[run_sample_omnidrive] RAW text_out:", text_out, flush=True)
    for qa in text_out:
        for ans in qa['A']:
            print("[run_sample_omnidrive] ANSWER:", ans, flush=True)


if __name__ == "__main__":
    main()
