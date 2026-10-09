"""Public UI never launches models, shells, GPU jobs or deployment services."""
import os


def validate_public_environment():
    private_options=[key for key,value in os.environ.items() if value and
        (key.startswith('IW_DAY1_') or key in ('IW_MODEL_CREATOR_AVOIDANCE','IW_PLANNER_REQUEST_CACHE_RELEASE'))]
    if private_options:
        raise ValueError('Private deployment/resume options are unavailable in this public entry: '+', '.join(sorted(private_options)))


def start_models():
    return ('公开版不提供自动模型启动。请自行配置已有的 HTTP LLM/Wan 服务；'
            '本请求未启动进程、加载模型或取得 GPU。')


def service_status(mode, get=None):
    """Read-only HTTP connectivity check; mock mode performs no network request."""
    from .presentation import service_status as check
    return check(mode, get=get)
