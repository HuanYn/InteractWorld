"""Narrow explicit-user sequence checks, never a plan generator or visual judge."""
import re


def explicit_ordered_calls(text):
    """Recognize only a complete, affirmative NPC-then-direction request."""
    match = re.fullmatch(
        r'\s*(?:先)?(?:走近|靠近|接近)\s*(Anna|Leo)\s*[，,。;；]?\s*'
        r'(?:然后|再|接着)\s*(?:观察|看看|看向|看)\s*(左边|右边|左侧|右侧)\s*[。.!！]?\s*',
        text, re.I)
    if match is None:
        return []
    direction = 'left' if match[2] in ('左边', '左侧') else 'right'
    return [['approach', match[1].lower(), None], ['inspect', direction, None]]


def ordered_coverage(required, observed):
    """Return the unsatisfied suffix without inserting or altering tool calls."""
    index = 0
    for call in observed:
        if index < len(required) and call == required[index]:
            index += 1
    return required[index:]
