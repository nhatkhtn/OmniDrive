"""Real-checkpoint smoke test for ``omnidrive_sae_adapter``.

This intentionally reuses the real sample/pipeline helpers from
``run_sample_omnidrive.py``.  It runs one native image-present inference,
compares direct next-token logits and greedy IDs with the hooked forward, and
checks that two changed image tensors reach Petr3D separately.  It is meant
for the prepared OmniDrive GPU environment; the CPU test lives under
``tests/``.
"""

import argparse
import copy
import os
import sys
import tempfile
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omnidrive_sae_adapter import OmniDriveActivationAdapter  # noqa: E402
from run_sample_omnidrive import (  # noqa: E402
    CHECKPOINT_PATH,
    LLM_PATH,
    REPO_ROOT,
    build_raw_sample,
    build_test_pipeline,
    check_env,
    register_single_question_pipeline,
)


def _patch_mmcv_stream_for_torch_27():
    """Keep the compatibility patch used by the native smoke script."""

    import torch.nn.parallel._functions as torch_functions
    import mmcv.parallel._functions as mmcv_functions

    original = torch_functions._get_stream

    def int_safe_get_stream(device):
        if isinstance(device, int):
            device = torch.device("cuda", device)
        return original(device)

    torch_functions._get_stream = int_safe_get_stream
    mmcv_functions._get_stream = int_safe_get_stream


def _build_model(config_path: str, checkpoint_path: str):
    import importlib

    import mmcv
    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import load_checkpoint
    from mmdet3d.models import build_model
    from projects.mmdet3d_plugin.models.dense_heads.llava_llama import (
        LlavaLlamaForCausalLM,
    )

    importlib.import_module("projects.mmdet3d_plugin")
    cfg = Config.fromfile(config_path)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.tokenizer = LLM_PATH
    cfg.model.lm_head = LLM_PATH
    cfg.model.save_path = tempfile.mkdtemp(prefix="omnidrive_sae_smoke_") + "/"

    print("[sae-smoke] building model", flush=True)
    # Petr3D's normal helper forces the 7B LLaVA checkpoint to CPU. Loading
    # its shards there exceeds the 60 GB smoke cgroup before MMDataParallel
    # can move the finished model to the B200. Scope the direct-GPU loader to
    # this smoke construction; it loads the same fp16 checkpoint tensors.
    original_from_pretrained = LlavaLlamaForCausalLM.from_pretrained

    def load_llava_on_gpu(cls, *model_args, **kwargs):
        kwargs = dict(kwargs)
        kwargs["device_map"] = {"": 0}
        kwargs.setdefault("low_cpu_mem_usage", True)
        kwargs.setdefault("offload_state_dict", True)
        return original_from_pretrained.__func__(cls, *model_args, **kwargs)

    LlavaLlamaForCausalLM.from_pretrained = classmethod(load_llava_on_gpu)
    try:
        detector = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    finally:
        del LlavaLlamaForCausalLM.from_pretrained
    print(f"[sae-smoke] loading checkpoint {checkpoint_path}", flush=True)
    # MMCV delegates local checkpoint reads to torch.load.  Memory-map the
    # 18 GB detector checkpoint so that only tensors being copied into the
    # model are faulted in; the normal loader otherwise materializes a second
    # full CPU copy alongside the already-built model.
    original_torch_load = torch.load

    def mmap_torch_load(*load_args, **load_kwargs):
        load_kwargs = dict(load_kwargs)
        load_kwargs["mmap"] = True
        load_kwargs.setdefault("weights_only", False)
        return original_torch_load(*load_args, **load_kwargs)

    torch.load = mmap_torch_load
    try:
        load_checkpoint(detector, checkpoint_path, map_location="cpu", strict=False)
    finally:
        torch.load = original_torch_load
    model = MMDataParallel(detector, device_ids=[0])
    model.eval()
    return model


def _assert_exact(name, left, right):
    if not torch.equal(left, right):
        delta = (left.float() - right.float()).abs().max().item()
        raise AssertionError(f"{name} changed (max absolute difference {delta:g})")


def _assert_batched_close(name, left, right):
    """Compare B=2 and B=1 paths, which may select different CUDA kernels."""

    try:
        torch.testing.assert_close(left, right, rtol=5e-2, atol=3e-1)
    except AssertionError as exc:
        delta = (left.float() - right.float()).abs().max().item()
        raise AssertionError(
            f"{name} drifted beyond the B=2 tolerance (max absolute difference {delta:g})"
        ) from exc
    return (left.float() - right.float()).abs().max().item()


def _duplicate_singleton_perception_data(perception_data):
    """Form the B=2 analogue of a native B=1 Petr3D input dictionary."""

    batched = {}
    for name, value in perception_data.items():
        if not isinstance(value, torch.Tensor) or value.ndim == 0 or value.shape[0] != 1:
            raise TypeError(
                "the smoke fixture expected a B=1 tensor for perception key "
                f"{name!r}, got {type(value).__name__} with shape "
                f"{getattr(value, 'shape', None)}"
            )
        batched[name] = value.repeat(2, *([1] * (value.ndim - 1)))
    return batched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample", default=os.path.join(REPO_ROOT, "data", "test_sample.json")
    )
    parser.add_argument(
        "--config",
        default=os.path.join(
            ROOT, "projects", "configs", "OmniDrive", "mask_eva_lane_det_vlm.py"
        ),
    )
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    args = parser.parse_args()

    check_env()
    if not torch.cuda.is_available():
        raise RuntimeError("OmniDrive SAE smoke test requires the allocated GPU")
    _patch_mmcv_stream_for_torch_27()

    from mmcv.parallel import collate
    from mmdet3d.datasets.pipelines import Compose

    register_single_question_pipeline()
    model = _build_model(args.config, args.checkpoint)
    raw_sample, _ = build_raw_sample(args.sample, REPO_ROOT)
    sample = Compose(build_test_pipeline())(raw_sample)
    data = collate([sample], samples_per_gpu=1)

    detector = model.module
    observed = {}
    original_extract = detector.extract_img_feat
    original_prepare = detector.prepare_location
    original_generate = detector.lm_head.generate

    def observe_extract(img):
        if "native_image" not in observed:
            observed["native_image"] = img.detach().clone()
        return original_extract(img)

    def observe_prepare(img_metas, **kwargs):
        if "img_metas" not in observed:
            observed["img_metas"] = img_metas
            observed["perception_data"] = {
                key: value
                for key, value in kwargs.items()
                # ``input_ids`` is a test-time question container.  It is
                # consumed only by Petr3D's generation loop, while this
                # helper reconstructs the perception path through the visual
                # query heads.
                if key not in {"img", "img_feats", "input_ids"}
            }
        return original_prepare(img_metas, **kwargs)

    def greedy_generate(*generate_args, **kwargs):
        inputs = kwargs.get("inputs")
        if inputs is None:
            if not generate_args:
                raise RuntimeError("native generate call did not provide inputs")
            inputs = generate_args[0]
        observed["input_ids"] = inputs.detach().clone()
        observed["vision_embeded"] = kwargs["images"].detach().clone()
        kwargs = dict(kwargs)
        kwargs["do_sample"] = False
        kwargs["num_beams"] = 1
        kwargs["max_new_tokens"] = args.max_new_tokens
        kwargs.pop("temperature", None)
        kwargs.pop("top_p", None)
        observed["baseline_ids"] = original_generate(*generate_args, **kwargs)
        return observed["baseline_ids"]

    detector.extract_img_feat = observe_extract
    detector.prepare_location = observe_prepare
    detector.lm_head.generate = greedy_generate
    try:
        with torch.no_grad():
            model(return_loss=False, rescale=True, **data)
    finally:
        detector.extract_img_feat = original_extract
        detector.prepare_location = original_prepare
        detector.lm_head.generate = original_generate

    required = {"native_image", "img_metas", "perception_data", "input_ids", "vision_embeded", "baseline_ids"}
    missing = required.difference(observed)
    if missing:
        raise RuntimeError(f"native smoke path did not expose {sorted(missing)}")

    input_ids = observed["input_ids"]
    vision_embeded = observed["vision_embeded"]
    adapter = OmniDriveActivationAdapter(
        detector,
        layer=args.layer,
        checkpoint=args.checkpoint,
        tokenizer=LLM_PATH,
        prompt_template="vicuna_v1",
    )

    with torch.no_grad():
        baseline = detector.lm_head(
            input_ids=input_ids,
            images=vision_embeded,
            use_cache=False,
            return_dict=True,
        )
        captured = adapter.capture(
            input_ids=input_ids,
            img=None,
            img_metas=observed["img_metas"],
            vision_embeded=vision_embeded,
            use_cache=False,
            return_dict=True,
        )

    _assert_exact("next-token logits", baseline.logits, captured.output.logits)
    selected_layer = adapter.decoder_layer()
    adapted_hook = selected_layer.register_forward_hook(lambda *_args: None)
    try:
        adapted_ids = original_generate(
            inputs=input_ids,
            images=vision_embeded,
            do_sample=False,
            num_beams=1,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )
    finally:
        adapted_hook.remove()
    _assert_exact("greedy token IDs", observed["baseline_ids"], adapted_ids)

    expected_d_model = int(getattr(detector.lm_head, "hidden_size", captured.activations.shape[1]))
    if captured.activations.ndim != 2 or captured.activations.shape[1] != expected_d_model:
        raise AssertionError(f"unexpected activation shape {tuple(captured.activations.shape)}")
    if captured.metadata["final_prompt_position"] is None:
        raise AssertionError("final prompt position was not recorded")

    # Verify that changed images invoke extraction afresh. Resetting the
    # temporal state before each standalone build mirrors a clean native pass.
    image_inputs = []
    original_extract = detector.extract_img_feat

    def track_extract(img):
        image_inputs.append(img.detach().clone())
        return original_extract(img)

    detector.extract_img_feat = track_extract
    try:
        changed_image = observed["native_image"].clone()
        changed_image.reshape(-1)[0] += 1e-3
        detector.test_flag = False
        with torch.no_grad():
            clean_visual = adapter.build_vision_embeded(
                observed["native_image"].clone(),
                observed["img_metas"],
                observed["perception_data"],
            )
        detector.test_flag = False
        with torch.no_grad():
            changed_visual = adapter.build_vision_embeded(
                # Petr3D squeezes a B=1 image tensor in place.  Keep the
                # source tensor 5-D so it can form the native B=2 input
                # below.
                changed_image.clone(),
                observed["img_metas"],
                observed["perception_data"],
            )
        if len(observed["img_metas"]) != 1:
            raise RuntimeError("the smoke fixture must expose exactly one image metadata item")
        batch_img_metas = [
            copy.deepcopy(observed["img_metas"][0]),
            copy.deepcopy(observed["img_metas"][0]),
        ]
        batch_perception_data = _duplicate_singleton_perception_data(
            observed["perception_data"]
        )
        batch_image = torch.cat((observed["native_image"], changed_image), dim=0)
        detector.test_flag = False
        with torch.no_grad():
            batch_visual = adapter.build_vision_embeded(
                batch_image,
                batch_img_metas,
                batch_perception_data,
            )
    finally:
        detector.extract_img_feat = original_extract

    if len(image_inputs) != 3 or torch.equal(image_inputs[0], image_inputs[1]):
        raise AssertionError("changed image did not reach a fresh Petr3D extraction")

    # The batch is the actual clean/changed perception workload used to supply
    # the VLM during SAE collection. Compare it to its B=1 components before
    # checking a capture hook at that exact B=2 shape.
    clean_visual_delta = _assert_batched_close(
        "clean B=2/B=1 vision embeddings", batch_visual[0], clean_visual[0]
    )
    changed_visual_delta = _assert_batched_close(
        "changed B=2/B=1 vision embeddings", batch_visual[1], changed_visual[0]
    )
    batch_input_ids = input_ids.expand(2, -1)
    with torch.no_grad():
        batch_baseline = detector.lm_head(
            input_ids=batch_input_ids,
            images=batch_visual,
            use_cache=False,
            return_dict=True,
        )
        batch_captured = adapter.capture(
            input_ids=batch_input_ids,
            img=None,
            img_metas=batch_img_metas,
            vision_embeded=batch_visual,
            use_cache=False,
            return_dict=True,
        )
    _assert_exact("batched next-token logits", batch_baseline.logits, batch_captured.output.logits)
    clean_batch_delta = _assert_batched_close(
        "clean B=2/B=1 logits", batch_captured.output.logits[0], captured.output.logits[0]
    )
    with torch.no_grad():
        changed_single = adapter.capture(
            input_ids=input_ids,
            img=None,
            img_metas=observed["img_metas"],
            vision_embeded=changed_visual,
            use_cache=False,
            return_dict=True,
        )
    changed_batch_delta = _assert_batched_close(
        "changed B=2/B=1 logits",
        batch_captured.output.logits[1],
        changed_single.output.logits[0],
    )
    activation_count = captured.activations.shape[0]
    clean_activation_delta = _assert_batched_close(
        "clean B=2/B=1 residuals",
        batch_captured.activations[:activation_count],
        captured.activations,
    )
    changed_activation_delta = _assert_batched_close(
        "changed B=2/B=1 residuals",
        batch_captured.activations[activation_count:],
        changed_single.activations,
    )

    duplicate_vision = torch.cat((clean_visual, clean_visual), dim=0)
    duplicate_ids = input_ids.expand(2, -1)
    batch_ids = original_generate(
        inputs=duplicate_ids,
        images=duplicate_vision,
        do_sample=False,
        num_beams=1,
        max_new_tokens=args.max_new_tokens,
        use_cache=True,
    )
    adapted_hook = selected_layer.register_forward_hook(lambda *_args: None)
    try:
        hooked_batch_ids = original_generate(
            inputs=duplicate_ids,
            images=duplicate_vision,
            do_sample=False,
            num_beams=1,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )
    finally:
        adapted_hook.remove()
    _assert_exact("batched greedy token IDs", batch_ids, hooked_batch_ids)
    if not (
        torch.equal(batch_ids[0], observed["baseline_ids"][0])
        and torch.equal(batch_ids[1], observed["baseline_ids"][0])
    ):
        raise AssertionError("batched greedy token IDs disagree with singleton decoding")

    print(
        "[sae-smoke] PASS",
        f"logits={tuple(baseline.logits.shape)}",
        f"activations={tuple(captured.activations.shape)}",
        f"final_prompt_position={captured.metadata['final_prompt_position']}",
        f"changed_visual_max_delta={(clean_visual - changed_visual).abs().max().item():g}",
        "batch_max_abs_delta="
        f"{max(clean_visual_delta, changed_visual_delta, clean_batch_delta, changed_batch_delta, clean_activation_delta, changed_activation_delta):g}",
        flush=True,
    )


if __name__ == "__main__":
    main()
