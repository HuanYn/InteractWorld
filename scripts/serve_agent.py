import argparse
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['mock','llm','wan','cinematic'],default='mock')
    parser.add_argument('--port',type=int,default=8873)
    parser.add_argument('--output-root',default=os.environ.get('IW_SESSION_ROOT','outputs/sessions'))
    parser.add_argument('--evidence-dir',help='Operator-selected result folder to display, not user-controlled')
    parser.add_argument('--visual-observer',action='store_true',help='Call VLM on rendered frames; does not turn mock footage into real footage')
    args=parser.parse_args()
    os.environ.setdefault('GRADIO_ANALYTICS_ENABLED','False')
    os.environ.setdefault('GRADIO_TEMP_DIR',str(Path(args.output_root).parent/'gradio-tmp'))
    from interactive.ui import build_ui
    app=build_ui(args.output_root,args.mode,args.evidence_dir,args.visual_observer)
    allowed=[str(Path(args.output_root).resolve())]
    if os.environ.get('IW_GAME_PRESENTATION')=='1' and os.environ.get('IW_GAME_ARCHIVE_MANIFEST'):
        import json
        archive=Path(os.environ['IW_GAME_ARCHIVE_MANIFEST']).resolve()
        # Explicit operator-listed media only; never expose weights/data roots.
        for item in json.loads(archive.read_text(encoding='utf-8'))['routes']:
            for value in item.get('videos',{}).values():
                media_path=Path(value).resolve()
                if media_path.is_file(): allowed.append(str(media_path))
    if os.environ.get('IW_REVIEWED_LIBRARY'):
        from interactive.reviewed_library import media_paths
        allowed.extend(media_paths(os.environ['IW_REVIEWED_LIBRARY']))
    if os.environ.get('IW_DELIVERY_RECORDINGS'):
        from interactive.delivery_recordings import recording_paths
        allowed.extend(str(p.resolve()) for p in recording_paths(os.environ['IW_DELIVERY_RECORDINGS']))
    if args.mode=='cinematic':
        manifest=Path(os.environ['IW_CINEMATIC_MANIFEST']).resolve()
        allowed.append(str(manifest.parent))
    if args.evidence_dir:
        allowed.extend(str(p.resolve()) for p in Path(args.evidence_dir).glob('wan-check-*') if p.is_dir())
        lab_sessions=Path(args.evidence_dir)/'real-lab-sessions'
        if args.mode=='wan' and lab_sessions.is_dir(): allowed.append(str(lab_sessions.resolve()))
        if args.mode=='wan':
            for name in ('door-framing-pilot','door-framing-six'):
                comparison=Path(args.evidence_dir)/name
                if comparison.is_dir(): allowed.append(str(comparison.resolve()))
    if os.environ.get('IW_FEEDBACK_DATA'):
        import json
        # Operator-owned manifest only, not paths supplied in a browser request.
        manifest=Path(os.environ['IW_FEEDBACK_DATA'])/'visual-candidates.jsonl'
        allowed.append(str((manifest.parent/'visual-reviews.jsonl').resolve()))
        if manifest.is_file():
            for line in manifest.read_text(encoding='utf-8').splitlines():
                if line:
                    path=Path(json.loads(line)['raw_video']).resolve()
                    if path.is_file(): allowed.append(str(path))
    if os.environ.get('IW_FEEDBACK_RUNS'):
        root=Path(os.environ['IW_FEEDBACK_RUNS'])
        for name in ('baseline','feedback','feedback-v2','feedback-v3','explicit-goal'):
            sessions=root/name/'sessions'
            if sessions.is_dir(): allowed.append(str(sessions.resolve()))
        allowed.append(str((root.parent/'continuation-20260927/interrupt-v1/sessions').resolve()))
    for env_key in ('IW_FEEDBACK_RUNS','IW_FEEDBACK_PREVIOUS_RUNS'):
        if not os.environ.get(env_key): continue
        result_root=Path(os.environ[env_key]).resolve()
        # These operator-owned session directories contain media/records, not
        # adapters, package caches or weights. Never expose the whole data root.
        sessions=list(result_root.glob('*/sessions'))
        sessions.extend(result_root/name/'sessions' for name in
            ('baseline','feedback','feedback-v2','feedback-v3','explicit-goal','live-validation'))
        for directory in sessions:
            resolved=directory.resolve()
            if resolved.is_relative_to(result_root): allowed.append(str(resolved))
    app.queue(max_size=8).launch(server_name='127.0.0.1',server_port=args.port,share=False,
        allowed_paths=allowed,show_error=False)


if __name__=='__main__': main()
