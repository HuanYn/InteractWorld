"""CPU fixtures, never nativeGPU/browser/quality measurements."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from training import completion_runtime_v2 as rt


class FakeBackend:
    branch_count=1;external_cfg_scale=1.;capacity=25
    def __init__(self):self.forward_count=0;self.actions=[];self.commits=[];self.closed=False
    def initialize(self,first,conditions):self.forward_count+=1;self.commits.append(first.clone())
    def velocity(self,**kw):
        self.forward_count+=1;self.actions.append(kw['action_block'].clone())
        return (kw['noisy_block']*.1+kw['action_block'].sum()*.001).to(torch.bfloat16)
    def commit(self,value,index):self.forward_count+=1;self.commits.append(value.clone())
    def close(self):self.closed=True


class FakeVAE:
    z_dim=48
    def __init__(self):self.model=self;self._feat_map=[None];self.clears=0;self.calls=[];self.bad=False
    def clear_cache(self):self.clears+=1;self._feat_map=[None]
    def decode_to_pixel(self,latent,*,use_cache,return_in_cpu):
        assert use_cache and not return_in_cpu
        self.calls.append(latent.shape[1]);frames=1 if self._feat_map[0] is None else 4*latent.shape[1]
        self._feat_map[0]=1
        return torch.zeros(1,frames+(1 if self.bad else 0),3,4,6)


def first():return torch.zeros(1,1,48,2,2,dtype=torch.bfloat16)
def conditions():return torch.zeros(1,5,4096,dtype=torch.bfloat16)
def engine(backend=None,max_chunks=8):return rt.NativeCausalActionEngine(backend or FakeBackend(),first(),conditions(),max_chunks=max_chunks)


def test_actual_four_core_calls_and_clean_commit_cpu():
    e=engine();e.initialize();out,row=e.generate_chunk(action_id='live-J',action_bits=[1,0,0,0,0,1,0,0])
    assert out.shape==(1,3,48,2,2) and e.backend.forward_count==6
    assert len(set(row['noise_seeds']))==4 and row['denoising_forwards']==4 and row['clean_commit_forwards']==1
    assert torch.equal(e.history[:,0:1],first()) and not row['future_clean_latents_supplied']
    assert all(torch.equal(a[0,0],torch.tensor([1,0,0,0,0,1,0,0])) for a in e.backend.actions)
    e.generate_chunk(action_id='live-L',action_bits=[1,0,0,0,0,0,0,1]);assert e.history.shape[1]==7
    assert e.backend.actions[-1][0,0,7]==1 and e.backend.forward_count==11


@pytest.mark.parametrize('bad',[[],[0]*7,[0]*9,[2]*8,[False]*8,[float('nan')]*8])
def test_action_contract(bad):
    e=engine();e.initialize()
    with pytest.raises(rt.RuntimeContractError):e.generate_chunk(action_id='x',action_bits=bad)


def test_prime_once_chunk_bound_and_fullhistory_capacity():
    e=engine(max_chunks=1)
    with pytest.raises(rt.RuntimeContractError):e.generate_chunk(action_id='x',action_bits=[0]*8)
    e.initialize()
    with pytest.raises(rt.RuntimeContractError):e.initialize()
    e.generate_chunk(action_id='x',action_bits=[0]*8)
    with pytest.raises(rt.RuntimeContractError):e.generate_chunk(action_id='x',action_bits=[0]*8)
    with pytest.raises(rt.RuntimeContractError):engine(max_chunks=80)


def test_decode_continuous48_prime_once_three_to_twelve():
    v=FakeVAE();d=rt.ContinuousWanDecoder(v,height=4,width=6)
    with pytest.raises(rt.RuntimeContractError):d.decode_chunk(first().repeat(1,3,1,1,1))
    out,row=d.warmup(first());assert out.shape[1]==1 and row['excluded_from_steady_state']
    for unused in range(4):
        out,row=d.decode_chunk(first().repeat(1,3,1,1,1));assert out.shape[1]==12 and row['continuous_cache']
    assert v.clears==1 and v.calls==[1,3,3,3,3]
    with pytest.raises(rt.RuntimeContractError):d.warmup(first())
    d.close();d.close();assert v.clears==2


@pytest.mark.parametrize('channel',[16,32,None])
def test_decode_reject_wrong_vae(channel):
    v=FakeVAE();v.z_dim=channel
    with pytest.raises(rt.RuntimeContractError):rt.ContinuousWanDecoder(v)


def test_decode_reject_wrong_count_and_reset_cache():
    v=FakeVAE();d=rt.ContinuousWanDecoder(v,height=4,width=6);d.warmup(first());v.bad=True
    with pytest.raises(rt.RuntimeContractError):d.decode_chunk(first().repeat(1,3,1,1,1))
    v.bad=False
    old=v.decode_to_pixel
    def reset(*args,**kw):
        value=old(*args,**kw);v._feat_map=[1];return value
    v.decode_to_pixel=reset
    with pytest.raises(rt.RuntimeContractError):d.decode_chunk(first().repeat(1,3,1,1,1))


def records(tmp_path):
    directory=tmp_path/'jobs'/'actual-ode80';directory.mkdir(parents=True)
    ckpt=directory/'checkpoint.pt';ckpt.write_bytes(b'CPU fixture not a tensor checkpoint')
    config=directory/'config.yaml';config.write_text('foo: fixture\n')
    def write(name,value):
        path=directory/(name+'.json');path.write_text(json.dumps(value));return dict(path=str(path),sha256=rt.file_sha256(path))
    request=dict(job_id=directory.name,mode='completion_ode_v2_train');rp=write('request',request)
    result=dict(job_id=directory.name,mode=request['mode'],request_sha256=rp['sha256'],status='completed',
        execution=dict(stage=rt.STUDENT_STAGES[0],step=80,checkpoint_path=str(ckpt)))
    op=write('result',result)
    settle=dict(status='settled',outcome='completed',reservation_id='actual-lease',charged_gpu_seconds=123.)
    sp=write('settlement',settle)
    ledger=tmp_path/'budget.json';ledger.write_text(json.dumps(dict(version=1,reservations={'actual-lease':
        dict(status='settled',charged_seconds=123.,settlement=settle,context=dict(job_id=directory.name,
        job_directory=str(directory),request_sha256=rp['sha256']))})))
    binding=dict(schema='completion_runtime_actual_student_v2',sampling_steps=4,external_cfg_scale=1.,latent_channels=48,
        producer_job_id=directory.name,endpoint_step=80,records=dict(request=rp,result=op,settlement=sp),
        checkpoint=dict(path=str(ckpt),bytes=ckpt.stat().st_size,mtime_ns=ckpt.stat().st_mtime_ns),
        config=dict(path=str(config),sha256=rt.file_sha256(config)))
    return binding,ledger


def test_actual_small_record_binding_yaml_not_checkpoint_success(tmp_path):
    binding,ledger=records(tmp_path)
    result=rt.validate_student_binding(binding,ledger_path=ledger,fixture_root=tmp_path)
    assert result['records_verified'] and not result['full_checkpoint_schema_verified'] and not result['gpu_profile_verified']
    Path(binding['checkpoint']['path']).unlink()
    with pytest.raises(rt.RuntimeContractError):rt.validate_student_binding(binding,ledger_path=ledger,fixture_root=tmp_path)


@pytest.mark.parametrize('kind',['request','result','settlement','config'])
def test_binding_changed_pin_failclosed(tmp_path,kind):
    binding,ledger=records(tmp_path);pin=binding['config'] if kind=='config' else binding['records'][kind]
    Path(pin['path']).write_text('{}')
    with pytest.raises(rt.RuntimeContractError):rt.validate_student_binding(binding,ledger_path=ledger,fixture_root=tmp_path)


def test_registry_no_paper_timings_or_zero_unavailable_values():
    rows=rt.runtime_rows();assert len(rows)==11
    for row in rows:
        assert row['sampling_steps']==4 and row['external_cfg_scale']==1 and row['latent_channels']==48
        assert all(v is None for v in row['measurements'].values())
        if any('fp8' in v or 'mxfp' in v for v in row['components']):assert row['status']=='hardware_unsupported'
    assert 'evictedhistory' in next(r for r in rows if r['row_id']=='project_local_kv7')['semantic_change']


def test_exact_kv_tensor_only_estimator():
    a=rt.kv_storage_bytes(layers=30,heads=24,head_dim=128)
    b=rt.kv_storage_bytes(layers=30,heads=24,head_dim=128,window_latents=61)
    assert a['video_kv_bytes']==30*2*7*390*24*128*2 and b['video_kv_bytes']>a['video_kv_bytes']*8
    assert a['excludes_weights_vae_activations_allocator']


class LocalBackbone:
    independent_first_frame=True;num_frame_per_block=3;use_relative_rope=False;patch_size=(1,2,2)
    def __init__(self):
        self.local_attn_size=-1;self.sink_size=0
        self.blocks=[NS(self_attn=NS(local_attn_size=-1,sink_size=0,max_attention_size=99))]
    def __call__(self,value,*,kv_cache,current_start,**unused):
        length=value.shape[2];end=current_start+length
        for row in kv_cache:
            old=int(row['global_end_index']);local=int(row['local_end_index'])
            if end>old:local=min(7,local+end-old)
            row['global_end_index'].fill_(end);row['local_end_index'].fill_(local)
        return torch.zeros_like(value)


def test_explicit_local_position_eviction_fixture_and_restore():
    backbone=LocalBackbone();model=NS(model=backbone)
    def kv(holder,*unused):holder.kv_cache1=[dict(global_end_index=torch.tensor(0),local_end_index=torch.tensor(0))]
    def cross(holder,*unused):holder.crossattn_cache=[]
    cfg=NS(data=NS(height=32,width=32),model=NS(action_scale=1.))
    backend=rt.LocalWanPositiveKV(model,cfg,cache_initializers=(kv,cross))
    e=engine(backend,max_chunks=4);e.initialize()
    for index in range(4):e.generate_chunk(action_id=f'action{index}',action_bits=[0]*8)
    assert backend.position==13 and int(backend.kv[0]['global_end_index'])==13
    assert int(backend.kv[0]['local_end_index'])==7 and backend.evictions==2
    assert backend.forward_count==21 and backbone.local_attn_size==7
    e.close();assert backbone.local_attn_size==-1 and backbone.blocks[0].self_attn.max_attention_size==99


@pytest.mark.parametrize('window',[0,3,6,8,9])
def test_local_window_whole_blocks(window):
    with pytest.raises(rt.RuntimeContractError):rt.LocalWanPositiveKV(NS(model=LocalBackbone()),None,window_latents=window)


def test_cpu_import_does_not_initialize_cuda():assert not torch.cuda.is_initialized()


def test_native_call_sites_protect_real_mixed_bf16_base_fp32_lora(monkeypatch):
    from training.models.lora import LoRALinear
    layer=LoRALinear(torch.nn.Linear(48,48,bias=False).to(torch.bfloat16),rank=4,alpha=4)
    assert layer.lora_a.weight.dtype==torch.float32 and layer.base.weight.dtype==torch.bfloat16
    def call(value):return layer(value.movedim(2,-1)).movedim(-1,2)
    with pytest.raises(RuntimeError):call(first())  # Genuine mixed-dtype failure withoutAMP onCPU.
    # CPU autocast tests the actual call sites; this is not a CUDA availability claim.
    monkeypatch.setattr(rt,'native_bf16_autocast',lambda device:torch.autocast('cpu',dtype=torch.bfloat16))
    class Mixed(FakeBackend):
        def initialize(self,value,conditions):call(value);super().initialize(value,conditions)
        def velocity(self,**kw):self.forward_count+=1;return call(kw['noisy_block'])
        def commit(self,value,index):call(value);super().commit(value,index)
    e=engine(Mixed());e.initialize();e.generate_chunk(action_id='test',action_bits=[0]*8)
    assert e.backend.forward_count==6 and layer.lora_a.weight.dtype==torch.float32


def test_actual_model_loader_and_builder_fail_before_missing_checkpoint_cuda():
    with pytest.raises(rt.RuntimeContractError):rt.load_actual_student_model({},torch.device('cpu'),lambda:None)
    with pytest.raises(rt.RuntimeContractError):rt.student_binding_from_job(Path('/missing'))
    assert not torch.cuda.is_initialized()


def native_gate_records(tmp_path):
    binding,ledger=records(tmp_path);directory=tmp_path/'jobs'/'native-probe';directory.mkdir()
    gate=dict(schema='completion_runtime_native_gate_v2',producer_job_id=binding['producer_job_id'],
        local_window_latents=None,runtime_source_sha256=rt.file_sha256(rt.__file__),
        checkpoint_schema_verified=True,installed_causality_verified=True,vae48_continuous_verified=True,
        peak_reserved_bytes=17*2**30)
    def write(name,value):
        path=directory/f'{name}.json';path.write_text(json.dumps(value));return dict(path=str(path),sha256=rt.file_sha256(path))
    request=dict(schema='completion_runtime_probe_request_v2',mode='completion_runtime_v2_probe',
        job_id=directory.name,student_binding=binding,local_window_latents=None)
    rp=write('request',request)
    result=dict(mode=request['mode'],job_id=request['job_id'],status='completed',request_sha256=rp['sha256'],
        execution=dict(native_runtime_gate=gate))
    op=write('result',result)
    settlement=dict(status='settled',outcome='completed',reservation_id='native-lease',charged_gpu_seconds=40.)
    sp=write('settlement',settlement)
    book=json.loads(ledger.read_text());book['reservations']['native-lease']=dict(status='settled',charged_seconds=40.,
        settlement=settlement,context=dict(job_id=request['job_id'],job_directory=str(directory),request_sha256=rp['sha256']))
    ledger.write_text(json.dumps(book));gate['actual_profile_records']=dict(request=rp,result=op,settlement=sp)
    return gate,binding,ledger


def test_native_gate_requires_actual_same_student_completed_ledger_and_pins_cpu_fixture(tmp_path):
    gate,binding,ledger=native_gate_records(tmp_path)
    value=rt.validate_native_gate(gate,binding,ledger_path=ledger,local_window_latents=None,fixture_root=tmp_path)
    assert value['peak_reserved_bytes']==17*2**30 and not torch.cuda.is_initialized()


@pytest.mark.parametrize('key,value',[('installed_causality_verified',False),('peak_reserved_bytes',24*2**30),
    ('runtime_source_sha256','changed'),('local_window_latents',7)])
def test_native_gate_cannot_edit_pass_values_after_real_records(tmp_path,key,value):
    gate,binding,ledger=native_gate_records(tmp_path);gate[key]=value
    with pytest.raises(rt.RuntimeContractError):
        rt.validate_native_gate(gate,binding,ledger_path=ledger,local_window_latents=None,fixture_root=tmp_path)
