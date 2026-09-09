"""Action-stage scale validation does not require GPU or downloaded weights."""
import pytest

from training.config import ActionTeacherConfig
from training.causal_tf import CausalTeacherForcingConfig


@pytest.mark.parametrize("config_class", [ActionTeacherConfig, CausalTeacherForcingConfig])
@pytest.mark.parametrize("scale", [0, 0.03, 0.1, 1.0])
def test_valid_action_scale(scale, config_class):
    config = config_class()
    config.model.action_scale = scale
    config.validate()
    assert config.to_dict()["model"]["action_scale"] == scale


@pytest.mark.parametrize("config_class", [ActionTeacherConfig, CausalTeacherForcingConfig])
@pytest.mark.parametrize("scale", [-1, float("nan"), float("inf"), True, "0.03", None])
def test_invalid_action_scale(scale, config_class):
    config = config_class()
    config.model.action_scale = scale
    with pytest.raises(ValueError, match="action_scale"):
        config.validate()
