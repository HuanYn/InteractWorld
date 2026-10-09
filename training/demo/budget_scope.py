"""Explicit Creator budget amendment; legacy grants keep their original limits."""
from training.demo.contracts import require


CREATOR_50H_SCOPE = 'creator_50h_300gb_20260928'
CREATOR_50H_LIMIT = 50
CREATOR_SERVER_LIMIT = 300_000_000_000
CREATOR_100H_SCOPE = 'creator_100h_500gb_20261003'
CREATOR_100H_LIMIT = 100
CREATOR_100H_SERVER_LIMIT = 500_000_000_000
CREATOR_BUDGET_SCOPES = {
    CREATOR_50H_SCOPE: (CREATOR_50H_LIMIT, CREATOR_SERVER_LIMIT),
    CREATOR_100H_SCOPE: (CREATOR_100H_LIMIT, CREATOR_100H_SERVER_LIMIT),
}
PROJECT_STORAGE_LIMIT = 1_100_000_000_000
LEGACY_STORAGE_LIMITS = (300_000_000_000, 400_000_000_000, 600_000_000_000)


def authority_budget_limits(authority, scope):
    """A null global limit is valid only with this exact, explicit amendment."""
    require(scope in (None, *CREATOR_BUDGET_SCOPES) and authority.get('budget_scope') == scope,
            'authorization budget scope does not match the configured/leased scope')
    storage = authority.get('total_storage_bytes_limit', 300_000_000_000)
    if scope is None:
        require(authority.get('total_gpu_hours_limit') == 160,
                'legacy authorization GPU-hour scope differs')
        require(type(storage) is int and storage in LEGACY_STORAGE_LIMITS,
                'legacy authorization storage budget scope changed')
        return dict(total_gpu_hours_limit=160, creator_gpu_hours_limit=6,
                    current_server_storage_bytes_limit=100_000_000_000, total_storage_bytes_limit=storage)
    require('total_gpu_hours_limit' in authority and authority['total_gpu_hours_limit'] is None
            and authority.get('project_gpu_hours_limit_removed') is True,
            'Creator amendment requires explicit removal of the project-wide GPU-hour ceiling')
    creator_limit, server_limit = CREATOR_BUDGET_SCOPES[scope]
    require(type(authority.get('creator_gpu_hours_limit')) in (int, float)
            and authority['creator_gpu_hours_limit'] == creator_limit
            and type(authority.get('current_server_storage_bytes_limit')) is int
            and authority['current_server_storage_bytes_limit'] == server_limit,
            f'Creator amendment requires explicit cumulative{creator_limit}h and current-server{server_limit // 1_000_000_000}GB limits')
    require(type(storage) is int and storage in (*LEGACY_STORAGE_LIMITS, PROJECT_STORAGE_LIMIT),
            'Creator amendment storage ceiling exceeds the explicit project limits')
    require(scope != CREATOR_100H_SCOPE or storage == PROJECT_STORAGE_LIMIT,
            'Creator100h amendment requires the unchanged explicit project1.1TB storage ceiling')
    return dict(total_gpu_hours_limit=None, creator_gpu_hours_limit=creator_limit,
                current_server_storage_bytes_limit=server_limit, total_storage_bytes_limit=storage)
