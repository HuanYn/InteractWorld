"""Software semantics only; not measured VLM/visual accuracy."""
from interactive.models import Plan
from interactive.visual_contracts import event_contract,qualify_contract

def test_left_is_viewpoint_not_object():
    c=event_contract(Plan(player_intent='test',semantic_action='inspect',target='left'))
    assert c['id']=='camera_pan_left' and 'CAMERA' in c['requirements']['viewpoint_motion']

def test_search_requires_key_not_proximity():
    c=event_contract(Plan(player_intent='test',semantic_action='search',target='desk'))
    assert 'key_evidence' in c['requirements']
    r=qualify_contract({'status':'supported','criteria':{'search_activity':'supported','key_evidence':'uncertain','object_continuity':'supported'}},c)
    assert r['status']=='uncertain'

def test_missing_criteria_cannot_pass():
    c=event_contract(Plan(player_intent='test',semantic_action='inspect',target='right'))
    assert qualify_contract({'status':'supported'},c)['status']=='uncertain'

def test_supported_criteria_do_not_upgrade_overall_uncertain():
    c=event_contract(Plan(player_intent='test',semantic_action='inspect',target='left'))
    r=qualify_contract({'status':'uncertain','criteria':{k:'supported' for k in c['requirements']}},c)
    assert r['status']=='uncertain'
