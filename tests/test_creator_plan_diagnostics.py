"""A bounded repair gate, not an alternate planner or task-success evaluator."""
import copy
import json

import pytest

from training.creator.planner import diagnose_proposal, plan_request


def edit(key, start, end):
    return dict(op="replace_intervals", key=key,
                intervals=[dict(start_seconds=start, end_seconds=end)])


def proposal(edits, scope="all", **extra):
    return dict(status="ready", edit_scope=scope, edits=edits, **extra)


def check(text, model=None, previous=None, code=None, repairable=False):
    saved = copy.deepcopy((previous, model))
    diagnosed = diagnose_proposal(text, previous, model)
    assert diagnosed["plan"] == plan_request(text, previous, model)
    assert diagnosed["feedback"]["code"] == code
    assert diagnosed["feedback"]["repairable"] is repairable
    assert (previous, model) == saved
    assert set(diagnosed["feedback"]) <= {"code", "repairable", "message",
                                          "missing_keys", "extra_keys", "protected_keys"}
    return diagnosed


def test_valid_preflight_and_proposal_need_no_repair():
    text = "前4秒向右移动，然后低头2秒"
    check(text, code="no_proposal")
    good = proposal([edit("D", 0, 4), edit("K", 4, 6)])
    result = check(text, good, code="accepted")
    assert result["plan"]["status"] == "ready"


def test_real_missing_second_action_is_repairable_but_not_replaced():
    text = "前4秒向右移动，然后低头2秒"
    result = check(text, proposal([edit("D", 0, 4)]),
                   code="missing_actions", repairable=True)
    assert result["feedback"]["missing_keys"] == ["K"]
    assert result["plan"]["status"] == "clarify"
    assert result["plan"]["action_segments"] == []
    assert "start_seconds" not in json.dumps(result["feedback"])
    assert "action_segments" not in result["feedback"]


def test_empty_requested_action_is_missing_even_if_its_key_is_listed():
    empty = dict(op="replace_intervals", key="K", intervals=[])
    result = check("前4秒向右移动，然后低头2秒", proposal([edit("D", 0, 4), empty]),
                   code="missing_actions", repairable=True)
    assert result["feedback"]["missing_keys"] == ["K"]


def test_wrong_scope_also_reports_missing_action_without_correct_answer():
    result = check("前4秒向右移动，然后低头2秒",
                   proposal([edit("D", 0, 4)], scope="movement"),
                   code="scope_mismatch", repairable=True)
    assert result["feedback"]["missing_keys"] == ["K"]


def test_wrong_time_is_repairable_without_supplying_correct_intervals():
    result = check("第8到10秒抬头", proposal([edit("I", 7, 10)]),
                   code="timing_mismatch", repairable=True)
    assert set(result["feedback"]) == {"code", "repairable", "message"}
    assert not any(character.isdigit() for character in result["feedback"]["message"])


def test_protected_track_change_names_only_affected_track():
    previous = dict(action_segments=[dict(frames=240, keys=["W", "J"])])
    model = dict(edit_scope="camera", action_segments=[dict(frames=128, keys=["W"]),
                 dict(frames=32, keys=["W", "I"]), dict(frames=80, keys=["W"])])
    result = check("第8到10秒抬头，其他不要动", model, previous,
                   code="protected_track_changed", repairable=True)
    assert result["feedback"]["protected_keys"] == ["J"]


@pytest.mark.parametrize("text,code", [
    ("第0.1到1秒抬头", "request_ambiguous"),
    ("走到桥边，然后回头看山", "request_unsupported"),
    ("晚一点", "request_ambiguous"),
    ("前进，低头，向右移动", "request_ambiguous"),
    ("后半段前进，然后前半段抬头", "request_ambiguous"),
])
def test_unsupported_ambiguous_and_non_frame_requests_are_not_model_errors(text, code):
    check(text, proposal([edit("I", 0, 15)]), code=code)
    preflight = check(text, code=code)
    assert preflight["plan"]["status"] != "ready"


@pytest.mark.parametrize("previous", [
    {"action_segments": [{"frames": 239, "keys": ["W"]}]},
    {"status": "clarify", "action_segments": []},
    {"action_segments": [{"frames": 240, "keys": ["W"]}], "bad": 1},
    [],
])
def test_invalid_baseline_is_not_repairable(previous):
    check("第8到10秒抬头", proposal([edit("I", 8, 10)]), previous,
          code="invalid_baseline")


@pytest.mark.parametrize("status", ["clarify", "unsupported"])
def test_explicit_model_decline_is_not_turned_into_retry(status):
    model = dict(status=status, edit_scope="all", edits=[], explanation="需要澄清")
    result = check("第8到10秒抬头", model, code="model_declined")
    assert result["plan"]["status"] == status
    # Even an inconsistent decline containing actions is not permission to
    # coerce a refusal into an executable answer.
    model["edits"] = [edit("I", 8, 10)]
    result = check("第8到10秒抬头", model, code="model_declined")
    assert result["plan"]["status"] != "ready"


@pytest.mark.parametrize("model", [
    [], {"status": "ready"},
    proposal([edit("I", 8, 10)], extra=True),
    proposal([edit("I", 0.1, 1)]),
    proposal([edit("I", 8, 10)], goals=[1]),
])
def test_structure_error_is_repairable_only_when_request_itself_is_legal(model):
    check("第8到10秒抬头", model, code="invalid_structure", repairable=True)
    check("第0.1到1秒抬头", model, code="request_ambiguous")


def test_extra_typed_edit_and_missing_legacy_action_have_precise_codes():
    result = check("第8到10秒抬头", proposal([edit("I", 8, 10), edit("W", 0, 4)]),
                   code="unexpected_actions", repairable=True)
    assert result["feedback"]["extra_keys"] == ["W"]
    model = dict(edit_scope="all", action_segments=[dict(frames=64, keys=["D"]),
                                                    dict(frames=176, keys=[])])
    result = check("前4秒向右移动，然后低头2秒", model,
                   code="missing_actions", repairable=True)
    assert result["feedback"]["missing_keys"] == ["K"]


def test_duration_error_can_be_fixed_without_mutating_baseline():
    previous = plan_request("全程前进，后半段抬头")
    model = dict(previous)
    model["edit_scope"] = "camera"
    check("保留前进，缩短抬头", model, previous,
          code="duration_mismatch", repairable=True)


def test_request_gate_precedes_even_legacy_ready_external_proposal():
    # The historical external validator allows >2 untimed actions, while the
    # bounded request interpreter declines them. Diagnostics preserves that
    # old API result but never admits it to the automatic model repair loop.
    model = dict(edit_scope="all", action_segments=[dict(frames=240, keys=["W", "D", "K"])])
    result = check("前进，低头，向右移动", model, code="request_ambiguous")
    assert result["plan"]["status"] == "ready"
