"""Capture valid-text residuals from OmniDrive's native VLM forward."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch


# These values are defined in the vendored LLaVA implementation.  Keeping the
# constants here avoids importing mmcv/transformers merely to inspect the
# position mapping on a CPU machine.
IMAGE_TOKEN_INDEX = -200
DEFAULT_VISUAL_TOKEN_COUNT = 513


@dataclass
class OmniDriveCapture:
    """Native model output, selected activations, and position metadata."""

    output: Any
    activations: torch.Tensor
    metadata: Dict[str, Any]


def build_omnidrive_position_map(
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    *,
    visual_token_count: int = DEFAULT_VISUAL_TOKEN_COUNT,
    image_token_index: int = IMAGE_TOKEN_INDEX,
    padding_side: str = "right",
) -> Dict[str, Any]:
    """Mirror LLaVA's one-sentinel, visual-token expansion for position maps."""

    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        raise ValueError("input_ids must be a rank-2 tensor shaped [batch, sequence]")
    if visual_token_count <= 0:
        raise ValueError("visual_token_count must be positive")
    if padding_side not in {"left", "right"}:
        raise ValueError("padding_side must be 'left' or 'right'")

    batch_size, raw_width = input_ids.shape
    if attention_mask is None:
        valid_mask = torch.ones(
            (batch_size, raw_width), dtype=torch.bool, device=input_ids.device
        )
    else:
        if not isinstance(attention_mask, torch.Tensor) or attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids")
        valid_mask = attention_mask.to(dtype=torch.bool)

    # The native preparation path eventually creates a right-padded expanded
    # batch.  Build all row-local mappings first, then pad the metadata masks
    # to that same maximum length.
    raw_valid_positions: List[List[int]] = []
    raw_text_positions: List[List[int]] = []
    post_text_positions: List[List[int]] = []
    visual_positions: List[List[Tuple[int, int]]] = []
    post_lengths: List[int] = []
    final_raw_positions: List[int] = []
    final_post_positions: List[int] = []

    input_cpu = input_ids.detach().cpu()
    mask_cpu = valid_mask.detach().cpu()
    for row in range(batch_size):
        valid_raw = torch.nonzero(mask_cpu[row], as_tuple=False).flatten().tolist()
        valid_tokens = [int(input_cpu[row, pos].item()) for pos in valid_raw]
        image_offsets = [
            pos for pos, token in enumerate(valid_tokens) if token == image_token_index
        ]
        if len(image_offsets) != 1:
            raise ValueError(
                "OmniDrive SAE capture requires exactly one image sentinel per "
                f"example; row {row} has {len(image_offsets)}"
            )

        cursor = 0
        row_raw_text: List[int] = []
        row_post_text: List[int] = []
        row_visual: List[Tuple[int, int]] = []
        for compact_pos, raw_pos in enumerate(valid_raw):
            if compact_pos in image_offsets:
                row_visual.append((cursor, cursor + visual_token_count))
                cursor += visual_token_count
            else:
                row_raw_text.append(raw_pos)
                row_post_text.append(cursor)
                cursor += 1

        if not row_raw_text:
            raise ValueError(f"row {row} has no text positions")
        raw_valid_positions.append(valid_raw)
        raw_text_positions.append(row_raw_text)
        post_text_positions.append(row_post_text)
        visual_positions.append(row_visual)
        post_lengths.append(cursor)
        final_raw_positions.append(row_raw_text[-1])
        final_post_positions.append(row_post_text[-1])

    max_post_length = max(post_lengths)
    post_attention_mask = torch.zeros((batch_size, max_post_length), dtype=torch.bool)
    text_position_mask = torch.zeros(
        (batch_size, max_post_length), dtype=torch.bool
    )
    padded_text_positions: List[List[int]] = []
    padded_visual_positions: List[List[Tuple[int, int]]] = []
    padded_final_post_positions: List[int] = []
    for row, (length, positions) in enumerate(zip(post_lengths, post_text_positions)):
        offset = max_post_length - length if padding_side == "left" else 0
        post_attention_mask[row, offset : offset + length] = True
        padded_positions = [offset + position for position in positions]
        text_position_mask[row, padded_positions] = True
        padded_text_positions.append(padded_positions)
        padded_visual_positions.append(
            [(offset + start, offset + end) for start, end in visual_positions[row]]
        )
        padded_final_post_positions.append(offset + final_post_positions[row])

    return {
        "raw_attention_mask": mask_cpu,
        "post_attention_mask": post_attention_mask,
        "text_position_mask": text_position_mask,
        "raw_valid_positions": raw_valid_positions,
        "raw_text_positions": raw_text_positions,
        "selected_sequence_indices": padded_text_positions,
        "visual_positions": padded_visual_positions,
        "post_sequence_lengths": post_lengths,
        "final_raw_prompt_positions": final_raw_positions,
        "final_prompt_sequence_indices": padded_final_post_positions,
        "visual_token_count": visual_token_count,
        "image_token_index": image_token_index,
        "padding_side": padding_side,
    }


class OmniDriveActivationAdapter:
    """Capture layer-``L`` residuals while leaving native outputs unchanged."""

    def __init__(
        self,
        petr3d: Any,
        layer: int,
        *,
        checkpoint: Optional[str] = None,
        tokenizer: Optional[str] = None,
        prompt_template: Optional[str] = "vicuna_v1",
        visual_token_count: int = DEFAULT_VISUAL_TOKEN_COUNT,
        image_token_index: int = IMAGE_TOKEN_INDEX,
    ) -> None:
        if not isinstance(layer, int) or isinstance(layer, bool) or layer < 0:
            raise ValueError("layer must be a non-negative integer")
        if visual_token_count <= 0:
            raise ValueError("visual_token_count must be positive")
        self.petr3d = petr3d
        self.layer_index = layer
        self.checkpoint = checkpoint
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        self.visual_token_count = visual_token_count
        self.image_token_index = image_token_index
        self._capture_module_path: Optional[str] = None

    @property
    def lm_head(self) -> Any:
        try:
            return self.petr3d.lm_head
        except AttributeError as exc:
            raise AttributeError("petr3d must expose a loaded lm_head") from exc

    @property
    def capture_module_path(self) -> str:
        return self._capture_module_path or (
            f"petr3d.lm_head.model.layers[{self.layer_index}]"
        )

    def _decoder_layers(self) -> Any:
        """Resolve native or PEFT-wrapped LLaMA decoder layers."""

        lm_head = self.lm_head

        def finish(layers: Any, named_path: str) -> Any:
            try:
                layer_count = len(layers)
            except TypeError as exc:
                raise AttributeError("resolved decoder layers are not indexable") from exc
            self._capture_module_path = (
                f"petr3d.lm_head.{named_path}[{self.layer_index}]"
            )
            if self.layer_index >= layer_count:
                raise IndexError(
                    f"layer {self.layer_index} is unavailable; model has {layer_count} layers"
                )
            return layers

        get_base_model = getattr(lm_head, "get_base_model", None)
        if callable(get_base_model):
            base_model = get_base_model()
            base_decoder = getattr(base_model, "model", None)
            base_layers = getattr(base_decoder, "layers", None)
            if base_layers is not None:
                named_path = None
                named_modules = getattr(lm_head, "named_modules", None)
                if callable(named_modules):
                    for name, module in named_modules():
                        if module is base_layers:
                            named_path = name
                            break
                if named_path is None:
                    named_path = "base_model.model.model.layers"
                return finish(base_layers, named_path)

        for named_path in (
            "model.layers",
            "base_model.model.model.layers",
            "base_model.model.layers",
        ):
            owner = lm_head
            for part in named_path.split(".")[:-1]:
                owner = getattr(owner, part, None)
                if owner is None:
                    break
            layers = getattr(owner, "layers", None) if owner is not None else None
            if layers is not None:
                return finish(layers, named_path)

        raise AttributeError(
            "OmniDrive LLaVA model must expose decoder layers directly or through "
            "lm_head.get_base_model()"
        )

    def _decoder_layer(self) -> Any:
        layers = self._decoder_layers()
        return layers[self.layer_index]

    def decoder_layer(self) -> Any:
        """Return the exact live layer used by the capture hook."""

        return self._decoder_layer()

    def _padding_side(self) -> str:
        side = (
            getattr(getattr(self.lm_head, "config", None), "tokenizer_padding_side", None)
            or "right"
        )
        if side not in {"left", "right"}:
            raise ValueError(f"unsupported LLaVA tokenizer_padding_side: {side!r}")
        return side

    def _reset_native_test_state_once(self) -> None:
        """Mirror ``Petr3D.forward_test``'s temporal reset behavior."""

        if getattr(self.petr3d, "test_flag", True):
            return
        for head_name in ("pts_bbox_head", "map_head"):
            head = getattr(self.petr3d, head_name, None)
            reset_memory = getattr(head, "reset_memory", None)
            if callable(reset_memory):
                reset_memory()
        if hasattr(self.petr3d, "test_flag"):
            self.petr3d.test_flag = True

    def build_vision_embeded(
        self,
        img: torch.Tensor,
        img_metas: Sequence[Mapping[str, Any]],
        perception_data: Optional[Mapping[str, Any]] = None,
    ) -> torch.Tensor:
        """Run Petr3D's native perception path and return visual tokens."""

        if img is None:
            raise ValueError("img is required to recompute OmniDrive vision tokens")
        data = dict(perception_data or {})
        data["img"] = img
        data["img_feats"] = self.petr3d.extract_img_feat(img)

        self._reset_native_test_state_once()
        location = self.petr3d.prepare_location(img_metas, **data)
        self.petr3d.forward_roi_head(location, **data)
        pos_embed = self.petr3d.position_embeding(data, location, img_metas)

        if not getattr(self.petr3d, "with_pts_bbox", False):
            raise RuntimeError("OmniDrive vision path has no pts_bbox_head")
        _, det_query = self.petr3d.pts_bbox_head(img_metas, pos_embed, **data)

        if not getattr(self.petr3d, "with_map_head", False):
            raise RuntimeError("OmniDrive vision path has no map_head")
        _, map_query = self.petr3d.map_head(img_metas, pos_embed, **data)

        vision_embeded = torch.cat([det_query, map_query], dim=1)
        if vision_embeded.ndim != 3:
            raise RuntimeError(
                "Petr3D visual queries must have shape [batch, tokens, hidden], "
                f"got {tuple(vision_embeded.shape)}"
            )
        if vision_embeded.shape[1] != self.visual_token_count:
            raise RuntimeError(
                "unexpected OmniDrive visual-token count: expected "
                f"{self.visual_token_count}, got {vision_embeded.shape[1]}"
            )
        return vision_embeded

    def _native_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        vision_embeded: torch.Tensor,
        *,
        image_sizes: Optional[Any] = None,
        **forward_kwargs: Any,
    ) -> Any:
        kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            images=vision_embeded,
        )
        if image_sizes is not None:
            kwargs["image_sizes"] = image_sizes
        kwargs.update(forward_kwargs)
        return self.lm_head(**kwargs)

    @staticmethod
    def _ensure_batch_tensor(value: torch.Tensor, name: str) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if value.ndim != 2:
            raise ValueError(f"{name} must be rank 1 or 2")
        return value

    def capture(
        self,
        input_ids: torch.Tensor,
        img: Optional[torch.Tensor],
        img_metas: Sequence[Mapping[str, Any]],
        *,
        perception_data: Optional[Mapping[str, Any]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        image_sizes: Optional[Any] = None,
        vision_embeded: Optional[torch.Tensor] = None,
        **forward_kwargs: Any,
    ) -> OmniDriveCapture:
        """Capture layer-``L`` residuals at valid text positions."""

        input_ids = self._ensure_batch_tensor(input_ids, "input_ids")
        if attention_mask is not None:
            attention_mask = self._ensure_batch_tensor(attention_mask, "attention_mask")
        if img is not None:
            vision_embeded = self.build_vision_embeded(img, img_metas, perception_data)
        elif vision_embeded is None:
            raise ValueError("provide img so vision tokens are recomputed, or vision_embeded")
        assert vision_embeded is not None
        if not isinstance(vision_embeded, torch.Tensor) or vision_embeded.ndim != 3:
            raise ValueError("vision_embeded must have shape [batch, tokens, hidden]")
        if vision_embeded.shape[0] != input_ids.shape[0]:
            raise ValueError("vision_embeded and input_ids batch sizes must match")
        if vision_embeded.shape[1] != self.visual_token_count:
            raise ValueError(
                f"vision_embeded has {vision_embeded.shape[1]} tokens; "
                f"expected {self.visual_token_count}"
            )

        position_map = build_omnidrive_position_map(
            input_ids,
            attention_mask,
            visual_token_count=self.visual_token_count,
            image_token_index=self.image_token_index,
            padding_side=self._padding_side(),
        )
        layer = self._decoder_layer()
        captured: List[torch.Tensor] = []

        def save_output(_module: Any, _inputs: Tuple[Any, ...], output: Any) -> None:
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(hidden, torch.Tensor):
                raise TypeError("decoder layer output does not contain a tensor")
            captured.append(hidden)

        hook = layer.register_forward_hook(save_output)
        try:
            output = self._native_forward(
                input_ids,
                attention_mask,
                position_ids,
                vision_embeded,
                image_sizes=image_sizes,
                **forward_kwargs,
            )
        finally:
            hook.remove()

        if len(captured) != 1:
            raise RuntimeError(
                f"expected one capture from {self.capture_module_path}, got {len(captured)}"
            )
        hidden = captured[0]
        if hidden.ndim != 3:
            raise RuntimeError(
                f"decoder residual must have shape [batch, sequence, d_model], got {tuple(hidden.shape)}"
            )
        if hidden.shape[0] != input_ids.shape[0]:
            raise RuntimeError("decoder residual batch size does not match input_ids")
        if hidden.shape[1] < max(position_map["post_sequence_lengths"]):
            raise RuntimeError(
                "decoder residual sequence is shorter than the multimodal position map"
            )

        selected_rows = [
            hidden[row, positions]
            for row, positions in enumerate(position_map["selected_sequence_indices"])
        ]
        activations = torch.cat(selected_rows, dim=0)
        d_model = int(hidden.shape[-1])

        metadata: Dict[str, Any] = dict(position_map)
        metadata.update(
            {
                "checkpoint": self.checkpoint,
                "tokenizer": self.tokenizer,
                "prompt_template": self.prompt_template,
                "layer": self.layer_index,
                "module_path": self.capture_module_path,
                "capture_point": "after decoder block, before final LLaMA norm",
                "capture_after_final_norm": False,
                "d_model": d_model,
                "selected_count": int(activations.shape[0]),
                "mask": position_map["text_position_mask"],
                "final_prompt_position": position_map["final_prompt_sequence_indices"][0]
                if len(position_map["final_prompt_sequence_indices"]) == 1
                else None,
                "final_raw_prompt_position": position_map["final_raw_prompt_positions"][0]
                if len(position_map["final_raw_prompt_positions"]) == 1
                else None,
            }
        )
        return OmniDriveCapture(output=output, activations=activations, metadata=metadata)


__all__ = [
    "DEFAULT_VISUAL_TOKEN_COUNT",
    "IMAGE_TOKEN_INDEX",
    "OmniDriveActivationAdapter",
    "OmniDriveCapture",
    "build_omnidrive_position_map",
]
