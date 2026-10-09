"""Explicit Oct8 Creator all-card supplement; never rebinds the old ledger.

CPU-validating this module is not permission to reserve. The separate sidecar
pins a complete immutable execution manifest and each actual job request.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import socket

from training.demo.budget_scope import CREATOR_100H_SCOPE, authority_budget_limits
from training.gpu_gate import SHARED_PROFILE

SCOPE = 'creator-completion-all-gpu-20261008-v1'
ROOT = Path('/DATA_DISK/InteractWorld/creator-runtime/abot-completion-20261007')
UUIDS = {
    0: 'GPU-00000000-0000-0000-0000-70d5a0d320b8',
    1: 'GPU-00000000-0000-0000-0000-cf150977eee9',
    2: 'GPU-00000000-0000-0000-0000-0d78f9cdb2ed',
    3: 'GPU-00000000-0000-0000-0000-0a13078b9aa3',
    4: 'GPU-00000000-0000-0000-0000-7d91c753c224',
}
MODES = {
    *('completion_ode_v2_'+s for s in ('probe','cache','merge','train','validate','eval')),
    *('completion_dmd_v2_'+s for s in ('profile','train','validate','eval')),
    'completion_service_v3_endpoint', 'completion_service_v4_endpoint',
    'completion_runtime_v1', 'completion_evaluation_v1',
    'completion_runtime_v2_probe', 'completion_runtime_v2_stream',
    'completion_evaluation_v4_60s',
    'completion_live_v5_probe', 'completion_live_v5_serve',
    'completion_live_noon_v1_probe', 'completion_live_noon_v1_serve',
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path, maximum=2_000_000):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= maximum,
            'supplement contract missing, symlinked or oversized')
    def unique(pairs):
        out = {}
        for key, value in pairs:
            require(key not in out, 'duplicate supplement contract key')
            out[key] = value
        return out
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite supplement JSON')))


def contracts(authority):
    pin = authority.get('job_contract_manifest')
    require(isinstance(pin, dict) and set(pin) == {'path','sha256'}, 'immutable job contract pin required')
    path = Path(pin['path'])
    require(path.is_absolute() and ROOT.resolve() in path.resolve().parents
        and re.fullmatch('[0-9a-f]{64}',pin['sha256']) and sha(path) == pin['sha256'],
        'job contract path/hash changed')
    value = read(path)
    require(value.get('schema') == 'creator_completion_all_gpu_contract_v1'
        and value.get('scope_root') == str(ROOT) and value.get('max_owned_active') == 4,
        'explicit completion root/four-card ceiling required')
    execution = value.get('execution_manifest')
    require(isinstance(execution,dict) and set(execution) == {'path','sha256'}, 'execution manifest pin required')
    manifest_path = Path(execution['path'])
    require(manifest_path.is_absolute() and ROOT.resolve() in manifest_path.resolve().parents
        and re.fullmatch('[0-9a-f]{64}',execution['sha256']) and sha(manifest_path) == execution['sha256'],
        'execution manifest changed')
    manifest = read(manifest_path)
    indices = manifest.get('allowed_gpu_indices')
    require(manifest.get('fixed_plan_complete') is True and isinstance(indices,list) and indices
        and all(type(i) is int and i in UUIDS for i in indices) and indices == sorted(set(indices))
        and type(manifest.get('max_owned_active')) is int and 1 <= manifest['max_owned_active'] <= 4
        and manifest.get('global_cap_gpu_seconds') == 360000
        and manifest.get('server_cap_bytes') == 500_000_000_000,
        'execution manifest cannot expand budget or omit its full fixed plan')
    rows = manifest.get('jobs')
    require(isinstance(rows,list) and rows and all(isinstance(row,dict) for row in rows), 'actual execution jobs required')
    by_id = {row.get('job_id'):row for row in rows}
    jobs = value.get('jobs')
    require(isinstance(jobs,dict) and len(by_id) == len(rows) and set(jobs) == set(by_id)
        and 0 < len(jobs) <= 256, 'all exact immutable job identities required')
    for identity, contract in jobs.items():
        require(isinstance(identity,str) and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,127}',identity)
            and isinstance(contract,dict)
            and set(contract) == {'directory','mode','source_revision','max_seconds','gpu_index'},
            'invalid exact job contract')
        row = by_id[identity]
        require(all(contract[k] == row.get(k) for k in ('directory','mode','max_seconds','gpu_index','source_revision')),
            'job contract differs from immutable execution node')
        directory = Path(contract['directory'])
        require(directory.is_absolute() and ROOT.resolve() in directory.resolve().parents
            and directory.name == identity and contract['mode'] in MODES
            and type(contract['max_seconds']) is int and 1 <= contract['max_seconds'] <= 3600
            and type(contract['gpu_index']) is int and contract['gpu_index'] in indices
            and isinstance(contract['source_revision'],str) and contract['source_revision']
            and len(contract['source_revision']) <= 128,
            'job directory/mode/watchdog/card/revision outside finite scope')
    require(authority.get('allowed_job_modes') == {k:v['mode'] for k,v in jobs.items()},
        'sidecar job modes must equal the complete pinned job contracts')
    return jobs


def validate_authority(authority, *, record, gpu_index, parent_record=None, parent=None, parent_sha=None):
    require(authority.get('gpu_authorization_scope') == SCOPE
        and authority.get('status') == 'ACTIVE' and authority.get('project') == 'InterActWorld'
        and authority.get('until_user_stop') is True and authority.get('scope_root') == str(ROOT)
        and authority.get('max_owned_active') == 4 and authority.get('user_message') ==
        '先启动啊，抓紧时间啊，全部gpu都授权给你，评测什么用第三个gpu',
        'explicit new all-card Creator supplement required')
    original_path = Path(authority.get('budget_authority_record',''))
    require(original_path.is_absolute() and original_path.resolve() != Path(record).resolve()
        and (parent_record is None or original_path.resolve() == Path(parent_record).resolve()),
        'separate unchanged original budget authority path required')
    parent = read(original_path,65536) if parent is None else parent
    parent_sha = sha(original_path) if parent_sha is None else parent_sha
    limits = authority_budget_limits(authority, CREATOR_100H_SCOPE)
    require(parent.get('status') == 'ACTIVE' and parent.get('project') == 'InterActWorld'
        and parent.get('until_user_stop') is True and parent.get('gpu_authorization_scope') is None
        and authority.get('budget_authority_id') == parent.get('authorization_id')
        and authority.get('authorization_id') != parent.get('authorization_id')
        and isinstance(authority.get('authorization_id'),str) and authority['authorization_id']
        and authority.get('budget_authority_sha256') == parent_sha
        and authority_budget_limits(parent, CREATOR_100H_SCOPE) == limits,
        'parent hash/status/budget identity changed')
    old_host = parent.get('hosts',{}).get(SHARED_PROFILE,{})
    require(isinstance(old_host.get('gpus'),list) and len(old_host['gpus']) == 2
        and {c.get('gpu_index') for c in old_host['gpus']} == {1,2}, 'original GPU1/2 authority required')
    hosts = authority.get('hosts')
    require(isinstance(hosts,dict) and set(hosts) == {SHARED_PROFILE}, 'same shared server only')
    host = hosts[SHARED_PROFILE]
    require(set(host) == {'hostname','gpus'} and str(host['hostname']).lower() ==
        str(old_host.get('hostname','')).lower() == socket.gethostname().lower()
        and host['gpus'] == [dict(gpu_index=i,gpu_uuid=u,desktop_memory_max_exclusive=500) for i,u in UUIDS.items()]
        and type(gpu_index) is int and gpu_index in UUIDS, 'fixed all-card UUIDs and idle threshold required')
    contracts(authority)
    return limits,dict(gpu_index=gpu_index,gpu_uuid=UUIDS[gpu_index],
        desktop_memory_max_exclusive=500,hostname=host['hostname'])


def validate_context(authority, *, job_directory, job_id, request_sha256, max_seconds, job_mode=None, gpu_index=None):
    jobs = contracts(authority)
    require(job_id in jobs, 'job is absent from complete immutable manifest')
    contract = jobs[job_id]
    directory = Path(job_directory)
    require(directory.is_absolute() and directory.resolve() == Path(contract['directory']).resolve()
        and directory.name == job_id and directory.is_dir(), 'job directory differs from pinned node')
    request_path = directory/'request.json'
    request = read(request_path)
    require(sha(request_path) == request_sha256 and request.get('job_id') == job_id
        and all(request.get(k) == contract[k] for k in ('mode','source_revision'))
        and type(max_seconds) is int and 1 <= max_seconds <= contract['max_seconds']
        and request.get('max_seconds') == max_seconds
        and request.get('contract_max_seconds') == contract['max_seconds']
        and (job_mode is None or job_mode == contract['mode'])
        and (gpu_index is None or gpu_index == contract['gpu_index']),
        'actual request hash/mode/revision/watchdog/card differs from pinned node')
    return contract['mode']
