# Public game source boundary

The game branch adds portable interaction code to the shared video foundation.
Mock mode is asset-free; LLM mode requires a configured provider, and Wan mode
requires compatible original assets and a same-filesystem loopback video service.

Publication-specific changes are confined to this branch's copy:

- UI outputs default to `outputs/sessions`, not a private server directory.
- Private cluster auto-launch and fixed-session recovery entries fail closed.
- Optional handoff closure is a generic operator switch, not a copied private ledger.
- Asset relocation uses operator-provided paths while retaining lineage checks.
- The portable Wan service requires explicit paths/device/launch acknowledgement,
  a whole-process timeout and an idle timeout. These are not GPU allocation or
  shared-card locks; the caller must supply independent exclusive scheduling.

No GPU inference, training, new gameplay or visual acceptance was performed to
prepare this release. The original deployed application is unchanged.

Excluded: model weights, embeddings/caches, datasets, raw session exports, raw
request traces, private launch grants, accounting ledgers, fixed server paths,
SSH configuration, screenshots and the newly edited presentation movie.

Software tests use small tensors/test doubles. The real key-route rule ending
does not establish visually correct pickup, unlocking or passage through the
same door. The previous finite interactive deployment is closed, not an online
public generation endpoint.
