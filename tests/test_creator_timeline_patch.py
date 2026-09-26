"""Input-control contracts only: no model understanding or video-quality claims."""
import copy
from decimal import Decimal

import pytest

from training.creator.planner import expand_segments, plan_request
from training.creator.timeline_patch import compile_edits, seconds_to_frame


def edit(key="I", start=8, end=10):
    return dict(op="replace_intervals", key=key,
                intervals=[dict(start_seconds=start, end_seconds=end)])


def active(plan, key):
    return [index for index, row in enumerate(expand_segments(plan["action_segments"])) if key in row]


@pytest.mark.parametrize("value,expected", [(0, 0), (15, 240), ("8.0625", 129),
                                           (Decimal("0.125"), 2), (7.5, 120)])
def test_seconds_are_exact_frame_boundaries(value, expected):
    assert seconds_to_frame(value) == expected


@pytest.mark.parametrize("value", [True, None, [], -1, 15.01, 0.1, "0.0625000001",
                                    "0.062500000000000000000000000000000000000001",
                                    "1e-10000", "NaN", "Infinity", float("nan"), "bad"])
def test_seconds_reject_invalid_or_non_frame_aligned_values(value):
    with pytest.raises(ValueError):
        seconds_to_frame(value)


def test_sparse_edit_preserves_all_other_tracks_without_mutation():
    original = [("W", "I", "J")] * 240
    before = copy.deepcopy(original)
    rows, metadata = compile_edits([edit()], original)
    assert original == before
    assert all(set(row) - {"I"} == {"W", "J"} for row in rows)
    assert [i for i, row in enumerate(rows) if "I" in row] == list(range(128, 160))
    assert metadata == dict(schema_version=1, fps=16, total_frames=240,
                            edits=[edit()], protected_keys=["W", "A", "S", "D", "J", "K", "L"])


@pytest.mark.parametrize("edits", [
    [], [edit(), edit()], [dict(op="add", key="I", intervals=[])],
    [dict(op="replace_intervals", key="SPACE", intervals=[])],
    [dict(edit(), extra=True)], [edit(start=10, end=8)], [edit(start=8, end=8)],
    [edit(start=0.1, end=1)], [dict(edit(), intervals=[
        dict(start_seconds=8, end_seconds=10), dict(start_seconds=9, end_seconds=11)])],
    [dict(edit(), intervals=[dict(start_seconds=8, end_seconds=10, frames=32)])],
])
def test_patch_rejects_unknown_duplicate_or_ambiguous_operations(edits):
    with pytest.raises(ValueError):
        compile_edits(edits)


def test_patch_rejects_conflict_with_a_protected_track():
    with pytest.raises(ValueError, match="opposing"):
        compile_edits([edit()], [("K",)] * 240)


@pytest.mark.parametrize("text", ["第8到10秒抬头", "只在第8-10秒抬头", "第8至10秒抬头",
                                  "只在第 8–10 秒抬头", "请在8到10秒镜头抬头"])
def test_explicit_range_grammar_compiles_half_open_interval(text):
    plan = plan_request(text)
    assert plan["status"] == "ready", plan
    assert active(plan, "I") == list(range(128, 160))
    assert plan["edit_patch"]["edits"] == [edit()]
    assert plan_request("取消抬头", plan)["status"] == "ready"


@pytest.mark.parametrize("text", ["前5秒前进，然后抬头3秒", "前5秒前进，随后抬头3秒", "前5秒前进然后抬头3秒"])
def test_duration_sequence_uses_previous_end_and_leaves_remainder_empty(text):
    plan = plan_request(text)
    assert plan["status"] == "ready", plan
    assert active(plan, "W") == list(range(80))
    assert active(plan, "I") == list(range(80, 128))
    assert plan["action_segments"][-1] == dict(frames=112, keys=[])


def test_full_movement_and_timed_camera_are_concurrent():
    plan = plan_request("全程前进，8-10秒抬头")
    assert plan["status"] == "ready"
    assert active(plan, "W") == list(range(240))
    assert active(plan, "I") == list(range(128, 160))


def test_timed_camera_edit_retains_other_camera_controls_too():
    previous = dict(action_segments=[dict(frames=240, keys=["W", "I", "J"])])
    for text in ("保持前进不变，只在第8–10秒抬头，其他不要动", "只改镜头，第8到10秒抬头"):
        result = plan_request(text, previous)
        assert result["status"] == "ready", result
        assert result["edit_scope"] == "camera" and result["preserved"]
        assert active(result, "W") == active(result, "J") == list(range(240))
        assert active(result, "I") == list(range(128, 160))


@pytest.mark.parametrize("text", ["第-1到2秒抬头", "第8到16秒抬头", "第0.1到1秒抬头",
                                  "前14秒前进，然后抬头3秒", "前5秒前进，然后第2到4秒抬头",
                                  "第8到10秒轻轻抬头", "前进，第8到10秒抬头",
                                  "前5秒前进，抬头3秒", "第8到10秒抬头，速度加倍",
                                  "全程前进，然后抬头3秒", "第8到10秒取消抬头"])
def test_partial_unsupported_ambiguous_or_conflicting_timing_is_not_executable(text):
    result = plan_request(text)
    assert result["status"] != "ready" and result["action_segments"] == []


def test_typed_proposal_is_checked_against_text_not_its_explanation():
    proposal = dict(status="ready", edit_scope="all", edits=[edit()],
                    explanation="Model says correct", goals=["镜头抬头"])
    assert plan_request("第8到10秒抬头", proposal=proposal)["status"] == "ready"
    for bad in ([edit(start=7, end=10)], [edit(key="K")],
                [edit(), edit(key="W", start=0, end=15)]):
        result = plan_request("第8到10秒抬头", proposal=dict(proposal, edits=bad))
        assert result["status"] == "clarify" and result["action_segments"] == []


def test_patch_goals_are_rebuilt_from_all_compiled_tracks_not_model_prose():
    previous = plan_request("一直前进，后半段抬头")
    proposal = dict(status="ready", edit_scope="movement",
                    edits=[edit(key="W", start=1.25, end=4.5)],
                    goals=["前3.25秒前进", "镜头已经成功抬头"])
    result = plan_request("保持镜头不变，只在第1.25到4.5秒前进", previous, proposal)
    assert result["status"] == "ready"
    assert result["goals"] == [
        "控制输入目标（不代表画面已完成）：W 向前移动输入 [1.25,4.5) 秒",
        "控制输入目标（不代表画面已完成）：I 镜头抬头输入 [7.5,15) 秒",
    ]
    assert plan_request("取消前进", result)["status"] == "ready"


def test_legacy_exact_proposal_and_fallback_also_use_compiled_goals():
    previous = plan_request("一直前进，后半段抬头")
    expected = plan_request("保留前进，第8到10秒抬头", previous)
    proposal = dict(expected, goals=["后半段抬头"])
    result = plan_request("保留前进，第8到10秒抬头", previous, proposal)
    assert result["status"] == "ready"
    assert result["goals"] == expected["goals"] == [
        "控制输入目标（不代表画面已完成）：W 向前移动输入 [0,15) 秒",
        "控制输入目标（不代表画面已完成）：I 镜头抬头输入 [8,10) 秒",
    ]


def test_nonpatch_goals_are_unchanged_and_patch_cancellation_is_explicit():
    plain = dict(edit_scope="all", action_segments=[dict(frames=240, keys=["W"])],
                 goals=["模型原有目标"])
    assert plan_request("前进", proposal=plain)["goals"] == ["模型原有目标"]
    previous = plan_request("一直前进，后半段抬头")
    removal = dict(edit_scope="camera", edits=[dict(op="replace_intervals", key="I", intervals=[])])
    result = plan_request("取消抬头", previous, removal)
    assert any("W " in goal and "[0,15)" in goal for goal in result["goals"])
    assert any("I " in goal and "该键已取消" in goal for goal in result["goals"])


def test_legacy_proposal_must_match_precise_time_and_preserve_every_other_key():
    previous = dict(action_segments=[dict(frames=240, keys=["W", "J"])])
    good = plan_request("第8到10秒抬头", previous)
    assert plan_request("第8到10秒抬头", previous, good)["status"] == "ready"
    wrong = dict(edit_scope="camera", action_segments=[dict(frames=128, keys=["W"]),
                  dict(frames=32, keys=["W", "I"]), dict(frames=80, keys=["W"])])
    assert plan_request("第8到10秒抬头", previous, wrong)["status"] == "clarify"


@pytest.mark.parametrize("text,segments", [
    ("先前进，然后抬头", [dict(frames=120, keys=["I"]), dict(frames=120, keys=["W"])]),
    ("全程前进，后半段抬头", [dict(frames=120, keys=["W"]), dict(frames=120, keys=["I"])]),
    ("前半段抬头", [dict(frames=120, keys=[]), dict(frames=120, keys=["I"])]),
    ("后半段前进，前半段抬头", [dict(frames=120, keys=["W"]), dict(frames=120, keys=["I"])]),
])
def test_legacy_proposal_cannot_reverse_or_drop_requested_temporal_semantics(text, segments):
    result = plan_request(text, proposal=dict(edit_scope="all", action_segments=segments))
    assert result["status"] == "clarify" and result["action_segments"] == []


def test_half_modifiers_bind_to_their_own_action():
    result = plan_request("后半段前进，前半段抬头")
    assert result["action_segments"] == [dict(frames=120, keys=["I"]), dict(frames=120, keys=["W"])]
    assert plan_request("先后半段前进，然后前半段抬头")["status"] == "clarify"
    assert plan_request("先前进，然后全程抬头")["status"] == "clarify"


@pytest.mark.parametrize("text", ["抬头，其他不要动", "只改镜头抬头，其他不要动"])
def test_other_unchanged_preserves_same_category_controls_even_without_seconds(text):
    previous = dict(action_segments=[dict(frames=240, keys=["W", "J"])])
    result = plan_request(text, previous)
    assert result["status"] == "ready" and result["preserved"]
    assert result["action_segments"] == [dict(frames=240, keys=["W", "I", "J"])]
    assert "J" in result["edit_patch"]["protected_keys"]
    dropped_j = dict(edit_scope="camera", action_segments=[dict(frames=240, keys=["W", "I"])])
    assert plan_request(text, previous, dropped_j)["status"] == "clarify"
    correct = dict(edit_scope="camera", edits=[edit(start=0, end=15)])
    assert plan_request(text, previous, correct)["action_segments"] == result["action_segments"]


@pytest.mark.parametrize("connector", ["再", "然后", "接着", "之后"])
def test_every_supported_sequential_connector_rejects_conflicting_halves(connector):
    text = f"后半段前进，{connector}前半段抬头"
    result = plan_request(text)
    assert result["status"] == "clarify" and result["action_segments"] == []
    reversed_plan = dict(edit_scope="all", action_segments=[dict(frames=120, keys=["I"]),
                                                            dict(frames=120, keys=["W"])])
    assert plan_request(text, proposal=reversed_plan)["status"] == "clarify"


def test_typed_cancel_and_existing_duration_edits_remain_compatible():
    previous = plan_request("一直前进，后半段抬头")
    proposal = dict(edit_scope="camera", edits=[dict(op="replace_intervals", key="I", intervals=[])])
    result = plan_request("保留前进，取消抬头", previous, proposal)
    assert result["status"] == "ready" and result["preserved"]
    assert result["action_segments"] == [dict(frames=240, keys=["W"])]


def test_no_competing_executable_schemas_and_nonready_has_no_edits():
    proposal = dict(edit_scope="all", edits=[edit()], action_segments=[dict(frames=240, keys=["I"])])
    assert plan_request("第8到10秒抬头", proposal=proposal)["status"] == "clarify"
    result = plan_request("第8到10秒抬头", proposal=dict(edit_scope="all", status="clarify", edits=[edit()]))
    assert result["status"] == "clarify" and result["action_segments"] == []
