"""CPU live orchestration/HTTP fixtures, not an actual browser/GPU profile."""
import json
import threading
import urllib.request

import pytest
import torch

from training import completion_runtime_v2 as rt
from training import completion_stream_v2 as st
from test_completion_runtime_v2 import FakeVAE,engine


class Clock:
    def __init__(self):self.now=1.
    def __call__(self):self.now+=.001;return self.now


def session(*,clock=None,max_chunks=2):
    return st.StreamSession(engine(max_chunks=max_chunks),rt.ContinuousWanDecoder(FakeVAE(),height=4,width=6),
        data_origin='cpu_fixture',max_chunks=max_chunks,clock=clock or Clock())


def attach(s):s.attach_browser('one-browser-clock',st.browser_build_hash())
def action(identity='J',bits=None,ms=1.):
    return dict(action_id=identity,bits=bits or [1,0,0,0,0,1,0,0],browser_clock_id='one-browser-clock',
        browser_action_ms=ms,origin='keyboard')
def ack(item,index=0,presented=100.):
    return dict(chunk_id=item['chunk_id'],frame_id=item['frame_ids'][index],action_id=item['action_id'],
        browser_clock_id='one-browser-clock',browser_presented_ms=presented)


def test_newest_live_actions_reach_each_new_generated_chunk_not_replay():
    s=session();attach(s);s.action(action('J'));s.prime();one=s.step();s.mark_sent(one)
    for index in range(12):s.acknowledge(ack(one,index,presented=100.+index))
    s.action(action('L',[1,0,0,0,0,0,0,1],2.));two=s.step();s.mark_sent(two)
    assert one['action_id']=='J' and two['action_id']=='L' and len(one['frames_jpeg'])==12
    assert s.engine.backend.actions[0][0,0,5]==1 and s.engine.backend.actions[-1][0,0,7]==1
    assert s.decoder.wrapper.calls==[1,3,3] and s.decoder.wrapper.clears==1
    events=[v['event'] for v in s.trace.events]
    assert events.index('action_received')<events.index('denoise_started')<events.index('decode_finished')<events.index('bytes_sent')
    assert not any('mp4' in v for v in one)


def test_presentation_ack_same_browser_clock_not_server_subtraction():
    s=session();attach(s);s.action(action(ms=10000.));s.prime();item=s.step();s.mark_sent(item)
    a=s.acknowledge(ack(item,presented=10123.));assert a['input_to_presented_ms']==123.
    result=s.results();assert result['distinct_acked_keyboard_actions']==1 and result['acknowledged_frames']==1
    assert all(v is None for v in result['metrics'].values())
    assert result['cpu_fixture_timing_not_project_measurements']['input_to_presented_p50_ms']==123.
    assert not result['native_gpu_with_browser_reported_ack'] and not result['realtime_claim_supported']


@pytest.mark.parametrize('field,value',[('action_id','wrong'),('frame_id','999:0'),('browser_clock_id','different'),
    ('chunk_id',999),('browser_presented_ms',-1),('browser_presented_ms',float('nan'))])
def test_invalid_ack_rejected(field,value):
    s=session();attach(s);s.action(action());s.prime();item=s.step();s.mark_sent(item);p=ack(item);p[field]=value
    with pytest.raises(rt.RuntimeContractError):s.acknowledge(p)


def test_ack_before_send_duplicate_and_negative_browser_latency_rejected():
    s=session();attach(s);s.action(action(ms=100.));s.prime();item=s.step()
    with pytest.raises(rt.RuntimeContractError):s.acknowledge(ack(item,presented=123.))
    s.mark_sent(item)
    with pytest.raises(rt.RuntimeContractError):s.acknowledge(ack(item,presented=99.))
    s.acknowledge(ack(item,presented=123.))
    with pytest.raises(rt.RuntimeContractError):s.acknowledge(ack(item,presented=124.))
    with pytest.raises(rt.RuntimeContractError):s.mark_sent(item)


def test_no_ack_means_no_visible_latency_or_browser_fps_even_if_pixels_generated():
    s=session();attach(s);s.action(action());s.prime();item=s.step();s.mark_sent(item)
    assert s.results()['cpu_fixture_timing_not_project_measurements']['browser_presented_fps'] is None
    assert s.results()['cpu_fixture_timing_not_project_measurements']['input_to_presented_p95_ms'] is None


def test_skipped_actions_reported_not_silently_counted_as_responses():
    s=session();attach(s);s.action(action('old',ms=1));s.action(action('new',ms=2));s.prime();item=s.step()
    assert item['action_id']=='new' and s.results()['superseded_or_unprocessed_action_ids']==['old']
    with pytest.raises(rt.RuntimeContractError):s.action(action('late',ms=1.5))
    with pytest.raises(rt.RuntimeContractError):s.action(action('new',ms=3.))


def test_finite_runtime_idle_and_chunk_bounds():
    clock=Clock();s=session(clock=clock,max_chunks=1);attach(s);s.prime();s.step()
    with pytest.raises(rt.RuntimeContractError):s.step()
    clock.now+=181
    with pytest.raises(rt.RuntimeContractError):s.check()


def test_real_minimal_browser_code_has_input_pixels_and_paint_ack_not_video():
    assert 'EventSource' in st.BROWSER and 'createImageBitmap' in st.BROWSER and 'drawImage' in st.BROWSER
    assert 'requestAnimationFrame(()=>requestAnimationFrame(r))' in st.BROWSER and "post('/ack'" in st.BROWSER
    assert 'performance.now()' in st.BROWSER and "post('/action'" in st.BROWSER
    assert '<video' not in st.BROWSER and '.mp4' not in st.BROWSER


def test_browser_build_binding_and_single_consumer():
    s=session()
    with pytest.raises(rt.RuntimeContractError):s.attach_browser('one','fakehash')
    attach(s)
    with pytest.raises(rt.RuntimeContractError):attach(s)


def test_structured_trace_preserves_previous_evidence(tmp_path):
    path=tmp_path/'runtime-events.jsonl';trace=st.Trace(path=path);trace.add('cpu_fixture',data_origin='cpu_fixture')
    assert json.loads(path.read_text())['data_origin']=='cpu_fixture'
    with pytest.raises(rt.RuntimeContractError):st.Trace(path=path)


def test_native_session_missing_actual_producer_fails_closed():
    with pytest.raises(rt.RuntimeContractError):
        st.StreamSession(None,None,data_origin='native_gpu')


def test_presentation_ack_backpressure_prevents_complete_future_pregeneration():
    s=session(max_chunks=2);attach(s);s.action(action('first',ms=1));s.prime();item=s.step();s.mark_sent(item)
    assert s.inflight_chunks()==1
    with pytest.raises(rt.RuntimeContractError):s.step()
    for index in range(11):s.acknowledge(ack(item,index,presented=100.+index))
    with pytest.raises(rt.RuntimeContractError):s.step()
    s.acknowledge(ack(item,11,presented=111.));assert s.inflight_chunks()==0
    s.action(action('latest-L',[1,0,0,0,0,0,0,1],2));last=s.step()
    assert last['action_id']=='latest-L' and not s.input_open
    with pytest.raises(rt.RuntimeContractError):s.action(action('too-late',ms=3))


def test_two_block_inflight_window_is_explicit_and_bounded():
    s=st.StreamSession(engine(max_chunks=3),rt.ContinuousWanDecoder(FakeVAE(),height=4,width=6),
        data_origin='cpu_fixture',max_chunks=3,max_inflight_chunks=2,clock=Clock())
    attach(s);s.prime();s.step();s.step()
    with pytest.raises(rt.RuntimeContractError):s.step()
    assert s.inflight_chunks()==2


def test_cpu_stdlib_http_tool_roundtrip_not_actual_browser():
    s=session();server=st.LocalStreamServer(s);worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
    base=f'http://127.0.0.1:{server.server_port}';suffix='?token='+s.token
    def post(path,value):
        request=urllib.request.Request(base+path+suffix,data=json.dumps(value).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=3) as response:return json.load(response)
    try:
        with urllib.request.urlopen(base+'/'+suffix,timeout=3) as response:
            html=response.read().decode();assert '__TOKEN__' not in html and s.token in html
        assert post('/hello',dict(browser_clock_id='one-browser-clock',browser_build_sha256=st.browser_build_hash()))['attached']
        assert post('/action',action())['accepted']=='J'
        s.prime();item=s.step();s.mark_sent(item)
        assert post('/ack',ack(item))['frame_id']=='0:0'
        with urllib.request.urlopen(base+'/results'+suffix,timeout=3) as response:
            result=json.load(response);assert result['data_origin']=='cpu_fixture' and result['metrics']['generation_fps'] is None
    finally:server.shutdown();server.server_close();worker.join(timeout=3)


def test_real_sse_bytes_and_presentation_release_latest_action_cpu_http():
    s=session(max_chunks=2);server=st.LocalStreamServer(s)
    worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
    producer=threading.Thread(target=s.produce,daemon=True);producer.start()
    base=f'http://127.0.0.1:{server.server_port}';suffix='?token='+s.token
    def post(path,value):
        request=urllib.request.Request(base+path+suffix,data=json.dumps(value).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=3) as response:return json.load(response)
    def event(stream):
        kind=stream.readline().decode().strip();data=json.loads(stream.readline().decode().removeprefix('data: '));stream.readline()
        assert kind=='event: chunk';return data
    try:
        post('/hello',dict(browser_clock_id='one-browser-clock',browser_build_sha256=st.browser_build_hash()))
        with urllib.request.urlopen(base+'/events'+suffix,timeout=5) as stream:
            first=event(stream);assert len(first['frames_jpeg'])==12 and s.inflight_chunks()==1
            assert post('/action',action('online-L',[1,0,0,0,0,0,0,1],2))['accepted']=='online-L'
            for index in range(12):post('/ack',ack(first,index,presented=100+index))
            second=event(stream);assert second['action_id']=='online-L' and second['chunk_id']==1
            for index in range(12):post('/ack',ack(second,index,presented=200+index))
        producer.join(timeout=3);assert not producer.is_alive() and s.failure is None
        assert s.results()['acknowledged_frames']==24 and s.engine.backend.closed
        assert all(v is None for v in s.results()['metrics'].values())
    finally:
        s.stop.set();server.shutdown();server.server_close();worker.join(timeout=3);producer.join(timeout=3)
