"""Append-only event reviews; requested actions are never relabelled as events."""
from datetime import datetime,timezone
import json
from pathlib import Path
import threading
import uuid

LOCK=threading.Lock()


def save_review(folder,candidate,verdict,event,start,end,note):
    if verdict not in ('supported','contradicted','uncertain'):
        raise ValueError('请选择符合、不符合或不确定。')
    if not str(note).strip(): raise ValueError('请说明画面证据，而不只复述请求。')
    if verdict=='supported' and not str(event).strip(): raise ValueError('请填写实际看见的事件。')
    if (start is None)!=(end is None): raise ValueError('时间起止必须同时填写或同时留空。')
    if start is not None and (float(start)<0 or float(end)<=float(start)):
        raise ValueError('事件结束时间必须大于开始时间，且开始时间不能小于零。')
    record={'review_id':uuid.uuid4().hex,'recorded_at':datetime.now(timezone.utc).isoformat(),
        'candidate_id':candidate['id'],'raw_video':candidate['raw_video'],
        'requested_plan':candidate.get('requested_plan'),'source_record':candidate.get('source_record'),
        'source_type':'user_submitted_visual_review','verdict':verdict,
        'actual_event':str(event).strip() or None,'start_seconds':start,'end_seconds':end,
        'evidence_note':str(note).strip(),'training_eligible':False,
        'remaining_gate':'duration/quality and source-rights checks before any training export',
        'rule_state_is_visual_truth':False}
    path=Path(folder)/'visual-reviews.jsonl'
    with LOCK,path.open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(record,ensure_ascii=False)+'\n')
    return record
