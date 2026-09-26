"""Launch the local-only Creator; generation still requires operator GPU authority."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deployment', type=Path, required=True)
    parser.add_argument('--provider-config', type=Path)
    parser.add_argument('--enable-visual-revision', action='store_true',
                        help='operator opt-in to one visual-feedback revision; off by default, observations remain uncalibrated')
    parser.add_argument('--serialize-model-and-video', action='store_true',
                        help='single-card deployment: reject overlapping model/video work immediately; GPU guard still applies')
    args = parser.parse_args()
    from training.demo.contracts import Deployment
    from training.demo.service import DemoService
    from training.creator.service import CreatorService, make_server
    provider = None
    if args.provider_config:
        from training.creator.providers import CommandProvider
        config = json.loads(args.provider_config.read_text(encoding='utf-8'))
        provider = CommandProvider(config['argv'], Path(config['runtime_root']), timeout_seconds=config.get('timeout_seconds', 300))
    demo = DemoService(Deployment.load(args.deployment))
    creator = CreatorService(demo, provider=provider, visual_revision_enabled=args.enable_visual_revision,
                             serialize_model_and_video=args.serialize_model_and_video)
    server = make_server(creator)
    print(f'InterActWorld-Creator: http://127.0.0.1:{server.server_address[1]}', flush=True)
    print('Planner: local model' if provider else 'Planner: transparent rule fallback; visual review: human only', flush=True)
    print('Visual revision: operator enabled' if args.enable_visual_revision else
          'Visual revision: disabled; uncalibrated model observations require human review', flush=True)
    print('Model/video serialization: enabled' if args.serialize_model_and_video else
          'Model/video serialization: off; operator must configure suitable GPU leases', flush=True)
    try:
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        demo.close()


if __name__ == '__main__':
    main()
