# Video research release: status and reproduction boundaries

This document describes the 2026-10-09 video source snapshot. The evidence snapshot confirms TF160, 64 ODE teacher trajectories, and an ODE20 endpoint. It does not confirm a completed genuine DMD update, a matched short/long DMD evaluation, or an accepted final streaming generator.

## Completed stages and remaining work

The project initially trained an action Adapter and LoRA on an official Wan2.2 base. Its historical Action R005 + window6 route generated selected 15-second clips by continuing 6 + 6 + 3 second windows from re-encoded terminal frames. Those outputs are useful asynchronous demonstrations; they do not establish a single continuous causal cache or realtime interaction.

The later research route uses a fresh project-trained action teacher, reference conditioning, causal Teacher Forcing, teacher ODE trajectories, a few-step student, then self-rollout distribution-matching distillation. Causal TF endpoints at 20/40/80/160 updates and ODE20 completed with checkpoint lineage. Visual shortcomings were retained; completed optimization steps alone do not establish useful action control or sufficient training.

The genuine DMD implementation keeps the frozen real score, trainable fake score, self-generated rollout, and generator VJP distinct. Native DMD attempts failed during causal/VJP checks before a complete optimizer endpoint; later attempts encountered CPU preparation or storage-accounting failures before GPU execution. No completed native DMD update was confirmed in the release snapshot. Sparse mask construction and token-chunked FFN are published implementation candidates, not a claim that the full DMD route now fits or succeeds on a 24 GB GPU.

The remaining evaluation requires two actual DMD arms with matched parent/data/noise/update counts, continuous 60-second generation, and all measured metrics including official HPSv3. Historical TF curves, short clips, CPU fixtures, or unmeasured/null HPS values do not complete that evaluation. A final qualified DMD generator, local history KV, continuous VAE, and human action-to-visible-frame measurements also remain outstanding.

## Measured TF timing

The table summarizes existing generation records, not new runs performed for publication. Conditions: two scenes, original versus left/right-swapped controls, four clips at each TF endpoint, seed 42, 832×480, 40-step Euler sampling, CFG 5, eight causal blocks, and 96 new frames plus the initial anchor. The four clips are different cases/conditions, not four independent training seeds or repeated benchmark trials.

| TF updates | Mean sampling work / s | Mean continuous decode + write / s | Mean combined work / s | Pooled generated FPS |
| ---: | ---: | ---: | ---: | ---: |
| 20 | 227.051 | 16.837 | 243.887 | 0.39362 |
| 40 | 227.746 | 16.699 | 244.444 | 0.39273 |
| 80 | 225.649 | 16.857 | 242.506 | 0.39587 |
| 160 | 226.899 | 16.921 | 243.820 | 0.39373 |

Combined means use the original unrounded values. Pooled generated FPS equals 384 new frames divided by the total sampling + decode/write seconds for all four clips at that endpoint. It excludes the retained anchor frame and separate shared loading. The clip's frame-index span is six seconds; a 97-frame container at 16 FPS has a nominal duration of 6.0625 seconds.

Sampling includes history/cache maintenance and checks inside the generation window. Decode/write includes continuous VAE decoding, RGB conversion, encoding/writing, and writer finalization; it is not pure VAE kernel time. Each job generated two clips before releasing the model and decoding both, so the sum of those work windows is not a clip's request-to-visible-frame delay. Shared loading, whole-job elapsed time, and billed GPU occupancy are different quantities. None is substituted for browser latency here.

The four endpoints use the same inference schedule. Their small timing differences do not show a training-length speedup. These short-horizon measurements are not extrapolated to 60 seconds, and no ODE/DMD inference timing is claimed.

## What can be reproduced from this branch

The CPU tests exercise ODE schedules and context visibility, DMD score windows and gradients, sparse-mask equivalence, FFN forward/backward equivalence, and streaming action/frame contracts using small tensors or fixtures. They do not load model weights or launch a training job.

The training modules preserve checkpoint schema validation, optimizer/RNG/cursor recovery, actual-parent bindings, and native causal/gradient checks. Some research wrappers are intentionally bound to a dated experimental layout, immutable job definitions, source hashes, endpoint requirements, and guarded execution. These are archival implementations, not ready-to-use authorizations. The original complete-comparison DMD entry requires ODE80; the actual shorter ODE20 route used separate bounded operator contracts that are not shipped as reusable deployment instructions.

For actual model work, obtain the model and dataset under their own terms, prepare the true aligned action/text/latent data and receipts, choose your own authorized hardware/resource policy, configure the source and data roots, and generate fresh manifests and hashes from the public source. Publicization changes paths and therefore source hashes. Do not copy an old hash, fabricate an endpoint, or relax causal/gradient checks to make an archived request pass.

`/DATA_DISK/InteractWorld` and `/WORKSPACE/InteractWorld` are placeholder roots. `GPU-00000000-0000-0000-0000-*` values are non-operational example identities. The existing configurable generic training path uses `INTERACTWORLD_ROOT`; the dated research wrappers additionally contain explicit experiment-root constants and require adaptation for a fresh deployment. No claim is made that setting one environment variable makes every archived wrapper launchable.

The repository does not distribute trained weights, original dataset videos, latent caches, private authorization documents, budget/lease records, remote-machine metadata, or raw operator logs. This publication does not run or modify an experiment.

## Research-source attribution

ABot-World and Wan attribution and included license notices are retained in the repository's `LICENSE`, `NOTICE`, and `THIRD_PARTY_NOTICES.md` files. Modified Wan code includes project action/reference/causal support; those changes do not make the upstream model architecture a project invention.

The genuine DMD implementation cites the formulation in [Causal-Forcing, pinned revision 8202903297918bcd12810fc4230a4db7fe065e64](https://github.com/thu-ml/Causal-Forcing/blob/8202903297918bcd12810fc4230a4db7fe065e64/model/dmd.py). The pinned project distributes its source under [Apache-2.0](https://github.com/thu-ml/Causal-Forcing/blob/8202903297918bcd12810fc4230a4db7fe065e64/LICENSE). The local code records the upstream revision and source-file digest, and adapts the objective to this project's action-conditioned native callbacks; upstream trained weights are not redistributed.
