"""CPU-only model-response parsing: no model, GPU, or repaired JSON evidence."""
import json

import pytest

from training.creator.model_worker import ModelResponseError, decode_model_response


@pytest.mark.parametrize('raw', [
    '  {"status":"ready","edits":[\n',
    '```json\n{"status":"ready"}\n```',
    'Here is the plan: {"status":"ready"}',
    '{"status":"ready"} trailing prose',
    '{"status":"ready","status":"clarify"}',
    '{"start_seconds":NaN}',
])
def test_malformed_response_retains_untouched_text_and_original_cause(raw):
    with pytest.raises(ModelResponseError) as caught:
        decode_model_response(raw)
    error = caught.value
    assert isinstance(error, ValueError)
    assert error.raw_text == raw
    assert isinstance(error.cause, ValueError)
    assert error.__cause__ is error.cause
    assert 'strict JSON' in str(error)


def test_json_decode_error_preserves_original_line_and_column():
    raw = '\n  {"status":"ready",\n"edits":]}'
    with pytest.raises(ModelResponseError) as caught:
        decode_model_response(raw)
    assert isinstance(caught.value.cause, json.JSONDecodeError)
    assert caught.value.cause.doc == raw
    assert caught.value.cause.lineno == 3
    assert caught.value.cause.colno == 9


def test_valid_json_is_decoded_without_changing_values_or_doing_schema_validation():
    raw = ' \n {"status":"clarify","explanation":"请明确时间","edits":[]} \n '
    assert decode_model_response(raw) == {
        'status': 'clarify', 'explanation': '请明确时间', 'edits': []}
    # Semantic schema validation remains in validate_plan / inspect handling.
    assert decode_model_response('[]') == []
