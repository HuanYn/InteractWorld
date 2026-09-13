# Local asynchronous demo

## Frozen demonstration profile (2026-09-13)

The selected demonstration is **Action R005 step1040 + window6 (6+6+3s)**.
See the [public frozen profile](../../docs/demo-window6-v1.md) for checkpoint
identity, output geometry and limitations. Select `--inference-mode window6`
explicitly in the preparation command below; the generic default remains
`chunked` for backward compatibility. Later causal/history-noise experiments do
not replace this profile. A recorded single-scenario run exists, but reliable
action control, general long-horizon stability and real-time performance are not
established. Model assets are not shipped. The separate [historical gallery](../../docs/demo-gallery.md)
contains selected generated clips for playback; that gallery is not this live model-backed UI.

This document describes the service protocol, not permission to run on any
machine. A deployment must satisfy its operator's current per-launch authority,
GPU occupancy, storage and budget rules; an old standing-grant schema is not a
new authorization. Leave the guard empty when only previewing the interface.

This UI queues **new self-trained model runs**. It is not an upstream video
gallery and does not claim real-time performance. Concrete backends support
`action_teacher_lora_v1`, `causal_teacher_forcing_v1` and `longforcing_lite_v1`.
Each stage must select its own matching adapter; Action output cannot stand in
for Causal/Long output. No production mock or sample-video fallback is provided.

## Operator setup (not browser input)

For Causal/Long, prepare a real rollout YAML using
`scripts/prepare_abot_demo_assets.py`. For the Action teacher, use the CPU-only
preparation module below. Pin an
immutable `step-*.pt` and its exact SHA256 in `lineage`; do not use mutable
`best.pt`. Keep the original training YAML/artifact hashes: a same-run extension
may have a larger saved `training.max_steps`, but it must not fabricate different
hashes. Both backends verify the actual saved checkpoint and original artifacts
before model construction. An Action checkpoint saved at step1040 with a saved
max_steps1040 can retain its original max_steps780 training YAML/hash; all other
serialized config fields and artifact hashes must still match exactly.

### Single-scene Action preparation

```bash
python -m training.demo.prepare_action \
  --training-config /path/to/original/action-training.yaml \
  --checkpoint /path/to/run/checkpoints/step-0001040.pt \
  --checkpoint-sha256 EXACT_CHECKPOINT_SHA256 \
  --initial-frame /path/to/demo/initial.png \
  --episode-id ACTUAL_HELD_OUT_EPISODE_ID \
  --action-input /path/to/preset/actions.npy \
  --initial-origin source_rgb \
  --seed 42 \
  --output /path/to/demo/rollout-action.yaml
```

The initial image must actually be RGB832x480. If reusing a PNG decoded from a
prior conditioning latent, declare `--initial-origin decoded_condition_rgb`.
The action input is `[240,8]` binary float32 `.npy`, or an existing frozen
`inputs.pt` from which only `actions` is extracted. The selected episode's
immutable static embedding and displayed text come from the original training
cache/receipt; no T5 re-encoding or future reference video is used. The command
checks the real checkpoint/cache/VAE on CPU and refuses to overwrite a config.

The explicit method `action_teacher_chunked_ar15s_ui_v1` runs five49-RGB-frame
segments, each with40 flow-Euler steps, shift5, BF16, noCFG. The actual initial
RGB is VAE-encoded now; each next segment encodes the previous generated floating
RGB endpoint before uint8 quantization. A single CPU generator seeded by the
request draws five successive noise tensors. Output frame0 is the submitted RGB,
followed by240 generated frames at16fps. This is a **new seeded UI generation**,
not the separate frozen-latent/frozen-noise regression protocol, native long
context, real-time generation, or proof of reliable action control. Original
training YAML and source checkpoints are never modified.

For an explicit asynchronous full-sequence alternative, add
`--inference-mode joint61` to the preparation command. The default remains
`chunked`. The joint method `action_teacher_joint61_15s_ui_v1` uses the same
self-trained noncausal Action checkpoint and40-step Euler sampler, but solves
all61 latent frames together with all240 action rows. It retains the ordered
12 future-noise slots from each of the same five seeded13-slot draws; the
first latent remains unchanged. After the DiT leaves the GPU, the VAE decodes
continuously with one retained cache, without generated-RGB re-encoding or
future ground-truth frames. The operator config records a separate adapter and
joint geometry; a browser cannot change the method. This is still a newly
seeded RGB-input UI protocol, not the frozen regression protocol or a claim
that the checkpoint was trained on61-frame contexts. It does not establish
general15-second stability, reliable control, or real-time performance.

The separate operator opt-in `--inference-mode window6` selects
`action_teacher_window6_15s_ui_v1`: 96/96/48 action rows drive25/25/13
latent-frame windows with the same40-step Euler solver. The five seeded noise
draws retain their60 ordered future slots, repartitioned24/24/12 without new
draws. Each subsequent window re-encodes the preceding raw floating RGB endpoint
in[-1,1], before uint8 conversion. Decoding97/97/49 frames and dropping the two
repeated conditioning frames yields97+96+48=241 output frames; output frame0
remains the submitted RGB. This is the selected inference-only demonstration profile,
not new training,
the frozen regression protocol, or a general quality/control pass. The default
`chunked` mode and browser request fields remain unchanged.

Example `/path/to/interactworld/demo/deployment.json`:

```json
{
  "rollout_config": "/path/to/interactworld/demo/rollout-action.yaml",
  "project_root": "/path/to/interactworld",
  "jobs_root": "/path/to/interactworld/demo/jobs",
  "host": "127.0.0.1",
  "port": 8765,
  "scene_count": 1,
  "max_pending": 2,
  "max_job_seconds": 900,
  "python_executable": "/path/to/interactworld/env/bin/python",
  "guard_command": []
}
```

```bash
python scripts/serve_abot_demo.py --deployment /path/to/interactworld/demo/deployment.json
```

An empty `guard_command` means **preview only**: submitting generation is
rejected. To enable actual GPU work, the operator supplies an absolute executable
argv list, for example `["/absolute/python", "/private/operator/demo_guard.py"]`.
No shell is invoked; browser requests cannot override this command or any GPU,
checkpoint, prompt, path, authority or budget field. Linux is required for the
GPU worker's process-group supervision. The service binds only `127.0.0.1`;
use a trusted SSH tunnel, not a public listener. The jobs root has one active
Linux service lock, one serial worker, and at most `max_pending` waiting jobs.

Each request snapshots the selected actual RGB initial frame, immutable scene
description, uint32 seed, and exactly `[240,8]` float32 actions. Key order is
`W,A,S,D,I,J,K,L`; UI arrows map to `I,J,K,L`. A full 241-frame/16fps rollout is
generated, followed by the current default input-header renderer. It never reads
reference future frames for user-generated actions. Actual VAE/base/checkpoint
assets must be available; Causal/Long also require their real T5 assets, whereas
Action uses its original static prompt cache without loading T5. The UI does
not replace missing models with media or a stub. Prompt editing remains closed.

## Private guard protocol

The fixed argv receives one JSON document on stdin and must return one JSON
document on stdout within **5 seconds**, with exit code0. Do not mix logs into
stdout. Stderr may contain operator diagnostics. Provider code and ledgers are
private and are not shipped in the public reproduction.

`reserve` input:

```json
{
  "operation": "reserve",
  "job_id": "unique-32-hex-id",
  "request_sha256": "exact-request-json-sha256",
  "max_seconds": 900,
  "project_root": "/path/to/interactworld",
  "job_directory": "/path/to/interactworld/demo/jobs/unique-32-hex-id"
}
```

Required `reserve` output fields:

- `status: "reserved"`, unique `reservation_id`, same `job_id` and `request_sha256`;
- `authorization_id`, `authorization_record` path, `authorization_sha256`, `profile`;
- `gpu_index: 0`, exact `gpu_uuid`, `desktop_memory_max_exclusive`;
- `gpu_hours_reserved >= (max_seconds + 30) / 3600`, including supervision/termination;
- `global_gpu_hours_upper <= 160`, including all completed usage and open reservations;
- `global_storage_bytes_upper <= 300000000000`;
- `start_before_unix`: absolute UNIX-seconds startup deadline, between now and now+900.

The startup lease is not a new user confirmation and must not refresh the real
standing-grant timestamp. The referenced authorization JSON must remain ACTIVE,
identify project `InterActWorld`, match `authorization_id`, set
`until_user_stop: true`, and preserve total limits160 GPU-hours/300GB. Its
`hosts[profile]` must match this machine's `hostname`, `gpu_index`, `gpu_uuid`, and
`desktop_memory_max_exclusive`. The worker itself performs the existing physical
GPU UUID/no-compute/desktop-memory gate before constructing any GPU model.

The provider must atomically allocate an exclusive local GPU sublease from the
operator's pre-reserved global allowance, and check storage/headroom. It must
never double-count or oversubscribe other hosts' reservations. This can use an
operator-reserved Web allowance in the main ledger plus an atomic local subledger;
the browser should not mutate the cross-host main ledger directly.

Every second the supervisor re-reads the active authorization/hash; every30s it
calls `check` with the same input context plus `lease` (the complete reservation).
Return `{"status":"allowed"}` only while budget, disk and authority remain
valid. Any exception, revocation, denied check or timeout ends only this worker's
owned process group, never another training process.

Finally `settle` receives the same context plus `lease`, `outcome`,
`elapsed_seconds` and `worker_returncode`. It must settle **idempotently by
reservation_id** and return `{"status":"settled"}`. The supervisor writes
`settlement-request.json` first and `settlement.json` only after confirmation.
Do not release an unresolved reservation silently; service/host crashes require
the operator to reconcile those receipts and verify process death. Restarted
queued/running jobs are marked failed, not replayed automatically.

## Artifacts and current verification boundary

Each unique job directory includes the frozen request/actions/initial/config,
worker log, lease, status transitions, generated `raw.mp4` and `inputs.mp4`,
lineage/input/video SHA receipts, and settlement records. The API serves only
allowlisted artifacts of completed jobs; paths, private lease and worker logs
are not a general file-serving surface. Host/Origin/CSRF checks reject cross-site
job submissions and DNS-rebinding Host headers. API requests are limited to64KB.

CPU tests cover the queue, exact frozen inputs, fail-closed stage/prompt/GPU
boundaries, HTTP origin/CSRF/path checks, actual Action/Causal worker wiring,
Action40-step timestep behavior and generated-RGB continuation/noise scheduling with
explicit test-only substitutes. They do not establish model quality, reliable
action control, GPU compatibility, or deployed-service completion. Actual GPU
generation and browser playback must still be checked with the selected final
self-trained checkpoint and operator guard.
