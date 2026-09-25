"""Pure CPU input contracts; these tests make no generated-video claims."""
import copy
import sys

import pytest

from training.creator.planner import compress_segments, expand_segments, plan_request


def previous_plan():
    return {"action_segments": [
        {"frames": 60, "keys": ["W"]},
        {"frames": 60, "keys": ["A"]},
        {"frames": 120, "keys": ["W", "I"]},
    ]}


def projection(plan, keys):
    return [set(row) & set(keys) for row in expand_segments(plan["action_segments"])]


def test_sequential_chinese_and_english_have_transparent_provenance():
    for text in ("先向前移动，后半段抬头", "first move forward, then look up"):
        result = plan_request(text)
        assert result["status"] == "ready"
        assert result["action_segments"] == [{"frames": 120, "keys": ["W"]}, {"frames": 120, "keys": ["I"]}]
        assert result["planner_kind"] == "rule_fallback"
        assert result["edit_scope"] == "all" and not result["preserved"]


def test_camera_shorten_preserves_every_movement_frame_and_source():
    previous = previous_plan()
    before = copy.deepcopy(previous)
    result = plan_request("保留前进，只缩短抬头", previous)
    assert result["status"] == "ready" and result["edit_scope"] == "camera"
    assert result["preserved"] and previous == before
    assert projection(result, "WASD") == projection(previous, "WASD")
    assert sum("I" in row for row in expand_segments(result["action_segments"])) == 60


def test_movement_edit_preserves_every_camera_frame():
    previous = previous_plan()
    result = plan_request("保留镜头，向右移动", previous)
    assert result["status"] == "ready" and result["edit_scope"] == "movement"
    assert result["preserved"]
    assert projection(result, "IJKL") == projection(previous, "IJKL")
    assert all("D" in row for row in expand_segments(result["action_segments"]))


def test_camera_directions_are_not_character_movement():
    for text, key in (("镜头向左", "J"), ("镜头向右", "L"), ("look left", "J"), ("camera right", "L")):
        result = plan_request(text)
        assert result["status"] == "ready"
        assert result["action_segments"] == [{"frames": 240, "keys": [key]}]
    assert plan_request("turn left")["status"] == "clarify"


def test_remove_and_increase_camera_duration_preserve_movement():
    previous = previous_plan()
    removed = plan_request("不要抬头", previous)
    increased = plan_request("增加抬头时间", previous)
    assert removed["status"] == increased["status"] == "ready"
    for result, count in ((removed, 0), (increased, 180)):
        assert sum("I" in row for row in expand_segments(result["action_segments"])) == count
        assert projection(result, "WASD") == projection(previous, "WASD")


def test_duration_only_is_ambiguous_unless_one_control_is_active():
    assert plan_request("减少时间", previous_plan())["status"] == "clarify"
    single = {"action_segments": [{"frames": 240, "keys": ["W"]}]}
    result = plan_request("减少时间", single)
    assert result["status"] == "ready" and result["edit_scope"] == "movement"
    assert result["action_segments"] == [{"frames": 120, "keys": ["W"]}, {"frames": 120, "keys": []}]


def test_unsupported_goals_cannot_be_hidden_by_a_valid_proposal():
    proposal = {"edit_scope": "all", "action_segments": [{"frames": 240, "keys": ["W"]}]}
    for text in ("拿钥匙", "开门", "跟 NPC 对话", "精确导航", "角色回头", "切换新场景", "pick up the key", "new scene"):
        result = plan_request(text, proposal=proposal)
        assert result["status"] == "unsupported" and result["action_segments"] == []


def test_unknown_or_conflicting_instructions_are_non_executable():
    for text in ("", "跳舞", "向前走三秒", "前进并后退", "缩短并增加抬头时间"):
        result = plan_request(text)
        assert result["status"] != "ready" and result["action_segments"] == []
    assert plan_request("保留前进，只缩短抬头")["status"] == "clarify"


def test_segment_roundtrip_canonicalizes_order_without_mutation():
    segments = [{"frames": 120, "keys": ["I", "W"]}, {"frames": 120, "keys": ["W", "I"]}]
    before = copy.deepcopy(segments)
    assert compress_segments(expand_segments(segments)) == [{"frames": 240, "keys": ["W", "I"]}]
    assert segments == before


def test_segment_validation_rejects_unknown_schema_conflicts_and_non_integer_frames():
    invalid = [
        [{"frames": True, "keys": []}], [{"frames": 240.0, "keys": []}],
        [{"frames": 240, "keys": ["W", "S"]}], [{"frames": 240, "keys": ["I", "K"]}],
        [{"frames": 240, "keys": ["J", "L"]}], [{"frames": 240, "keys": ["SPACE"]}],
        [{"frames": 240, "keys": ["W", "W"]}], [{"frames": 239, "keys": []}],
        [{"frames": 241, "keys": []}], [{"frames": 240, "keys": [], "extra": True}],
    ]
    for segments in invalid:
        with pytest.raises(ValueError):
            expand_segments(segments)


def test_external_proposal_preserved_flag_is_recomputed():
    previous = previous_plan()
    good = plan_request("保留前进，只缩短抬头", previous)
    proposal = {**good, "preserved": False, "planner_kind": "made_up"}
    result = plan_request("保留前进，只缩短抬头", previous, proposal)
    assert result["status"] == "ready" and result["preserved"]
    assert result["planner_kind"] == "external_proposal_validated"


def test_malicious_proposal_cannot_change_movement_or_expand_scope():
    previous = previous_plan()
    good = plan_request("保留前进，只缩短抬头", previous)
    for malicious in (
        {**good, "action_segments": [{"frames": 240, "keys": ["I"]}], "preserved": True},
        {**good, "edit_scope": "all", "preserved": True},
        {**good, "ignore_constraints": True},
        {**good, "action_segments": [{"frames": 240.0, "keys": ["W"]}]},
    ):
        result = plan_request("保留前进，只缩短抬头", previous, malicious)
        assert result["status"] == "clarify" and not result["preserved"]
        assert result["action_segments"] == []


def test_malicious_proposal_cannot_change_camera_in_movement_scope():
    previous = previous_plan()
    malicious = {"edit_scope": "movement", "preserved": True,
                 "action_segments": [{"frames": 240, "keys": ["D"]}]}
    assert plan_request("保留镜头，向右移动", previous, malicious)["status"] == "clarify"


def test_proposal_must_perform_requested_edit_and_previous_must_be_valid():
    previous = previous_plan()
    unchanged = {**previous, "edit_scope": "camera"}
    assert plan_request("缩短抬头", previous, unchanged)["status"] == "clarify"
    invalid_previous = {"action_segments": [{"frames": 200, "keys": ["W"]}]}
    assert plan_request("缩短抬头", invalid_previous)["status"] == "clarify"
    assert plan_request("保留镜头，抬头", previous)["status"] == "clarify"
    for text in ("只改镜头，前进", "只改移动，抬头", "edit camera only, move forward"):
        result = plan_request(text, previous)
        assert result["status"] == "clarify" and result["action_segments"] == []
    assert plan_request("只改镜头，向右看", previous)["edit_scope"] == "camera"


def test_simple_planner_module_has_no_heavy_imports_and_supports_overlay():
    # The planner itself must not import demo.contracts (which imports numpy).
    assert "torch" not in sys.modules["training.creator.planner"].__dict__
    result = plan_request("一直向前移动，后半段抬头")
    assert result["status"] == "ready"
    assert result["action_segments"] == [{"frames": 120, "keys": ["W"]}, {"frames": 120, "keys": ["W", "I"]}]
