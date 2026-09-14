"""CPU-only checks for the OmniDrive activation adapter."""

from types import SimpleNamespace

import torch
from torch import nn

from omnidrive_sae_adapter import (
    IMAGE_TOKEN_INDEX,
    OmniDriveActivationAdapter,
    build_omnidrive_position_map,
)


def test_position_map_excludes_visual_tokens_and_tracks_final_prompt():
    input_ids = torch.tensor(
        [
            [10, IMAGE_TOKEN_INDEX, 20, 30, 0],
            [40, 50, IMAGE_TOKEN_INDEX, 60, 0],
        ]
    )
    attention_mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 0]])

    mapping = build_omnidrive_position_map(
        input_ids, attention_mask, visual_token_count=3
    )

    assert mapping["raw_text_positions"] == [[0, 2, 3], [0, 1, 3]]
    assert mapping["selected_sequence_indices"] == [[0, 4, 5], [0, 1, 5]]
    assert mapping["final_raw_prompt_positions"] == [3, 3]
    assert mapping["final_prompt_sequence_indices"] == [5, 5]
    assert mapping["post_attention_mask"].tolist() == [[True] * 6, [True] * 6]
    assert mapping["text_position_mask"].tolist() == [
        [True, False, False, False, True, True],
        [True, True, False, False, False, True],
    ]


def test_position_map_tracks_left_padded_expanded_batches():
    input_ids = torch.tensor(
        [
            [0, 10, IMAGE_TOKEN_INDEX, 20, 30],
            [0, 0, 40, IMAGE_TOKEN_INDEX, 50],
        ]
    )
    attention_mask = torch.tensor([[0, 1, 1, 1, 1], [0, 0, 1, 1, 1]])

    mapping = build_omnidrive_position_map(
        input_ids,
        attention_mask,
        visual_token_count=3,
        padding_side="left",
    )

    assert mapping["selected_sequence_indices"] == [[0, 4, 5], [1, 5]]
    assert mapping["final_prompt_sequence_indices"] == [5, 5]
    assert mapping["post_attention_mask"].tolist() == [
        [True] * 6,
        [False, True, True, True, True, True],
    ]


class _FakeBlock(nn.Module):
    def forward(self, hidden):
        return (hidden + 1.0,)


class _FakeLlava(nn.Module):
    """A tiny native-shaped LLaVA model for output-preservation testing."""

    def __init__(self, width=4):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_FakeBlock()])
        self.proj = nn.Linear(width, 5, bias=False)

    def forward(self, input_ids, attention_mask=None, images=None, **kwargs):
        mask = (
            torch.ones_like(input_ids, dtype=torch.bool)
            if attention_mask is None
            else attention_mask.bool()
        )
        rows = []
        for row in range(input_ids.shape[0]):
            pieces = []
            for token in input_ids[row][mask[row]]:
                if int(token) == IMAGE_TOKEN_INDEX:
                    pieces.append(images[row])
                else:
                    pieces.append(
                        torch.full((1, images.shape[-1]), float(token.item()))
                    )
            rows.append(torch.cat(pieces, dim=0))
        max_len = max(row.shape[0] for row in rows)
        hidden = torch.stack(
            [
                torch.cat(
                    [
                        row,
                        torch.zeros((max_len - row.shape[0], row.shape[1])),
                    ]
                )
                for row in rows
            ]
        )
        hidden = self.model.layers[0](hidden)[0]
        return SimpleNamespace(logits=self.proj(hidden))


class _FakePeftWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = nn.Module()
        self.base_model.model = base_model

    def get_base_model(self):
        return self.base_model.model

    def forward(self, *args, **kwargs):
        return self.base_model.model(*args, **kwargs)


class _FakeHead:
    def __init__(self, count, width):
        self.count = count
        self.width = width

    def reset_memory(self):
        pass

    def __call__(self, img_metas, pos_embed, **data):
        value = data["img"].reshape(data["img"].shape[0], -1).mean(dim=1)
        return None, value[:, None, None].expand(-1, self.count, self.width)


class _FakePetr3D:
    with_pts_bbox = True
    with_map_head = True
    test_flag = False

    def __init__(self):
        self.lm_head = _FakeLlava()
        self.pts_bbox_head = _FakeHead(2, 4)
        self.map_head = _FakeHead(1, 4)

    def extract_img_feat(self, img):
        return img

    def prepare_location(self, img_metas, **data):
        return data["img"].new_zeros((data["img"].shape[0], 1, 1, 1))

    def forward_roi_head(self, location, **data):
        return {}

    def position_embeding(self, data, location, img_metas):
        return data["img"].new_zeros((data["img"].shape[0], 1, 4))


def test_capture_preserves_native_logits_and_argmax_ids():
    detector = _FakePetr3D()
    adapter = OmniDriveActivationAdapter(
        detector, layer=0, visual_token_count=3, checkpoint="fake.pth"
    )
    input_ids = torch.tensor([[10, IMAGE_TOKEN_INDEX, 20, 30]])
    image = torch.ones((1, 1))
    visual = adapter.build_vision_embeded(image, [{}])
    baseline = detector.lm_head(input_ids=input_ids, images=visual)
    captured = adapter.capture(
        input_ids=input_ids,
        img=None,
        img_metas=[{}],
        vision_embeded=visual,
    )

    torch.testing.assert_close(captured.output.logits, baseline.logits, rtol=0, atol=0)
    assert torch.equal(captured.output.logits.argmax(dim=-1), baseline.logits.argmax(dim=-1))
    assert captured.activations.shape == (3, 4)
    assert captured.metadata["d_model"] == 4
    assert captured.metadata["final_prompt_position"] == 5
    assert captured.metadata["checkpoint"] == "fake.pth"


def test_capture_resolves_peft_wrapped_decoder_layer():
    detector = _FakePetr3D()
    detector.lm_head = _FakePeftWrapper(detector.lm_head)
    adapter = OmniDriveActivationAdapter(detector, layer=0, visual_token_count=3)
    input_ids = torch.tensor([[10, IMAGE_TOKEN_INDEX, 20, 30]])
    visual = adapter.build_vision_embeded(torch.ones((1, 1)), [{}])

    captured = adapter.capture(
        input_ids=input_ids,
        img=None,
        img_metas=[{}],
        vision_embeded=visual,
    )

    assert captured.activations.shape == (3, 4)
    assert captured.metadata["module_path"] == (
        "petr3d.lm_head.base_model.model.model.layers[0]"
    )


def test_capture_batches_padded_prompts_against_singletons():
    detector = _FakePetr3D()
    adapter = OmniDriveActivationAdapter(detector, layer=0, visual_token_count=3)
    input_ids = torch.tensor(
        [
            [10, IMAGE_TOKEN_INDEX, 20, 30, 0],
            [40, IMAGE_TOKEN_INDEX, 50, 0, 0],
        ]
    )
    attention_mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]])
    visual = torch.stack(
        [
            torch.ones((3, 4)),
            torch.full((3, 4), 2.0),
        ]
    )

    captured = adapter.capture(
        input_ids=input_ids,
        img=None,
        img_metas=[{}, {}],
        attention_mask=attention_mask,
        vision_embeded=visual,
    )
    first = adapter.capture(
        input_ids=input_ids[:1, :4],
        img=None,
        img_metas=[{}],
        vision_embeded=visual[:1],
    )
    second = adapter.capture(
        input_ids=input_ids[1:, :3],
        img=None,
        img_metas=[{}],
        vision_embeded=visual[1:],
    )

    torch.testing.assert_close(
        captured.activations,
        torch.cat((first.activations, second.activations)),
    )
    assert captured.metadata["selected_sequence_indices"] == [[0, 4, 5], [0, 4]]
    assert captured.metadata["final_prompt_position"] is None
