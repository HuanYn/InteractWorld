import pytest
import torch

from scripts.cache_scene_static_prompts import encode_prompts, static_caption


def test_static_caption_does_not_select_narrative_or_motion_fields():
    assert static_caption({"scene_static": "  grassy hillside  ", "narrative": "pan right and walk"}) == "grassy hillside"


@pytest.mark.parametrize("caption", [None, "walk", {}, {"narrative": "pan right"}, {"scene_static": " "}, {"scene_static": ["forest"]}])
def test_missing_static_caption_fails_closed(caption):
    with pytest.raises(ValueError):
        static_caption(caption)


def test_encoder_receives_exact_static_text_and_keeps_episode_binding():
    seen = []
    def encoder(text):
        seen.append(text)
        return {"prompt_embeds": torch.ones(1, 512, 4096)}
    result = encode_prompts({"b": {"prompt": "stone road"}, "a": {"prompt": "forest"}}, encoder)
    assert seen == [["forest"], ["stone road"]]
    assert list(result) == ["a", "b"]
    assert all(t.dtype == torch.bfloat16 and t.device.type == "cpu" for t in result.values())
