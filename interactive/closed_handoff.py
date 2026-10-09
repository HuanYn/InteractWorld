"""Operator-selected read-only UI mode; no deployment or private ledger binding."""
import os

MODE = 'read_only'
REASON = '此入口已设为只读；可查看与导出已有记录，不启动模型或继续生成。'


def closed_reason():
    # Invalid nonempty settings also block actions instead of enabling work.
    return REASON if os.environ.get('IW_CLOSED_DEMO_HANDOFF') else None


def verify_closed_handoff():
    if not closed_reason():
        return None
    if os.environ['IW_CLOSED_DEMO_HANDOFF'] != MODE:
        raise ValueError('Set IW_CLOSED_DEMO_HANDOFF=read_only, or leave it unset')
    return {'mode': MODE, 'reason': REASON, 'new_gpu_authority': False}
