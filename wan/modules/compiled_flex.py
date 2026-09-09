# InterActWorld reproduction: adapted from ABot-World; see NOTICE.
"""Bounded, compiled-only FlexAttention for the fixed Wan window shapes."""

import torch
import torch._dynamo


# LongForcing uses 4/7/10/13/12-latent non-TF windows plus the 13-latent
# two-branch TF replay, in grad and no-grad modes.  The default eight entries
# cannot hold these variants.  Keep a finite allowance for their stride/mask
# guards; do not silently expand it if a new shape or guard exhausts the bound.
FLEX_RECOMPILE_LIMIT = 32


def compile_sparse_flex_attention(attention):
    """Keep sparse kernels or fail; never substitute eager dense attention.

    PyTorch 2.8 shares the compilation cache across a function's code object.
    Its limit is process-global, so patch it only while entering this compiled
    callable (including checkpoint recomputation), then restore the caller's
    settings.  The total accumulated-cache limit remains unchanged.
    """
    compiled = torch.compile(
        attention, dynamic=False, fullgraph=True,
        mode="max-autotune-no-cudagraphs",
    )

    def compiled_only(*args, **kwargs):
        config = torch._dynamo.config
        if config.accumulated_recompile_limit < FLEX_RECOMPILE_LIMIT:
            raise RuntimeError(
                "FlexAttention requires accumulated_recompile_limit >= "
                f"{FLEX_RECOMPILE_LIMIT}; refusing an eager fallback"
            )
        with config.patch(
            recompile_limit=FLEX_RECOMPILE_LIMIT,
            fail_on_recompile_limit_hit=True,
            suppress_errors=False,
        ):
            return compiled(*args, **kwargs)

    return compiled_only
