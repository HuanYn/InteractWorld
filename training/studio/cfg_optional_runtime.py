"""Opt-in CFG difference reuse for an existing bounded Studio segment.

This module does not grant GPU access, enable a UI capability, change a default,
or certify quality. The operator must qualify the option and bind the original
R005 source. Exact generation delegates to the existing QualityRuntime.
"""
from __future__ import annotations

import copy

from training.demo.contracts import require
from training.demo.latency_engine import _WanRuntime
from training.studio.cfg_delta_probe import CFGDeltaFlow, POLICY
from training.studio.quality_probe import QualityRuntime

CFG_EXECUTIONS = ('exact', 'delta_cached_v1')


class OptionalCFGRuntime(QualityRuntime):
    """Configure once before each run_segment; default execution is exact.

    The caller keeps authorization, source identity, the shared lease, watchdog,
    and request idempotency. A failed runtime must be closed, not reconfigured.
    No input-identity pin from a particular mountain/grass fixture is reused.
    """

    def configure(self, case, cfg_execution='exact'):
        require(type(cfg_execution) is str and cfg_execution in CFG_EXECUTIONS,
                'unsupported optional CFG execution mode')
        require(not getattr(self, '_optional_busy', False)
                and not getattr(self, '_optional_failed', False),
                'optional CFG runtime is busy or failed; close failed runtime')
        require(type(case) is dict and type(case.get('steps')) is int and case['steps'] == 40
                and case.get('solver', 'flow_euler') == 'flow_euler'
                and all(type(case.get(key)) in (int, float) and case[key] == value
                        for key, value in [('cfg_scale', 5), ('lora_strength', 1), ('action_scale', .03)]),
                'optional CFG runtime requires 40Euler / CFG5 / LoRA1 / action0.03')
        super().configure(case)
        self.cfg_execution = cfg_execution
        self.cfg_delta_trace, self.cfg_delta_stats = [], {}
        self._optional_busy = self._optional_failed = self._optional_generated = False
        self._optional_configured = True

    def generate(self, first, prompt, keys, noise, steps, *, fastpath=False):
        require(getattr(self, '_optional_configured', False)
                and not self._optional_busy and not self._optional_failed and not self._optional_generated,
                'configure once per Studio segment; no automatic CFG retry')
        require(type(steps) is int and steps == 40 and fastpath is False
                and self.cfg_scale == 5 and self.solver == 'flow_euler',
                'optional CFG requires exact40step schedule and disabled fastpath')
        require(tuple(first.shape) == (1, 1, 48, 30, 52) and noise.ndim == 5
                and noise.shape[0] == 1 and noise.shape[1] in (13, 25)
                and tuple(noise.shape[2:]) == (48, 30, 52)
                and tuple(keys.shape) == (1, (noise.shape[1] - 1) * 4, 8),
                'optional CFG supports only independent 832x480 3s/6s segments')
        self._optional_busy = self._optional_generated = True
        original, flow = self.model, None
        status = 'failed'
        try:
            if self.cfg_execution == 'exact':
                # Keep original CFG expression, execution order and precision.
                latent = super().generate(first, prompt, keys, noise, steps, fastpath=False)
                require(self.forward_calls == 80, 'exact CFG segment requires80model forwards')
            else:
                flow = CFGDeltaFlow(original,
                    self.negative.to(device='cuda', dtype=self.torch.bfloat16),
                    expected_latent_frames=noise.shape[1],
                    on_trace=getattr(self, 'trace_callback', None))
                self.model = flow
                # Do not call QualityRuntime.generate: that would wrap CFG twice.
                latent = _WanRuntime.generate(self, first, prompt, keys, noise, steps, fastpath=False)
                require(flow.completed_steps == len(flow.trace) == 40 and flow.forward_calls == 64
                        and flow.positive_calls == 40 and flow.negative_calls == 24,
                        'cached CFG requires40steps /64forwards /40positive /24negative')
            status = 'completed'
            return latent
        except BaseException:
            self._optional_failed = True
            raise
        finally:
            self.model = original
            self._optional_busy = False
            if flow is not None:
                self.forward_calls += flow.forward_calls
                self.cfg_delta_trace = copy.deepcopy(flow.trace)
                self.cfg_delta_stats = dict(completed_logical_steps=flow.completed_steps,
                    positive_forward_calls=flow.positive_calls, negative_forward_calls=flow.negative_calls,
                    cached_steps=sum(row['approximate'] and row['status'] == 'completed' for row in flow.trace),
                    peak_delta_cache_bytes=flow.peak_cache_bytes)
                flow.close()
            self.cfg_delta_stats.update(status=status, latent_frames=noise.shape[1],
                actual_model_forward_calls=self.forward_calls,
                expected_model_forward_calls=64 if self.cfg_execution == 'delta_cached_v1' else 80,
                cross_segment_reuse=False, cross_window_reuse=False, cache_scope='one_generate_call',
                cache_released_before_decode=flow is None or (flow.closed and flow.delta is None),
                model_restored=self.model is original)


def annotate_optional_cfg(report, runtime):
    """Add execution evidence AFTER quality_probe.annotate_report, including failure.

    The result records a requested approximation, not a quality/latency verdict.
    Exact mode removes any stale optional policy left on a reused report object.
    """
    require(getattr(runtime, 'cfg_execution', None) in CFG_EXECUTIONS,
            'configure optional CFG runtime before annotating a report')
    cached = runtime.cfg_execution == 'delta_cached_v1'
    report.update(cfg_execution=runtime.cfg_execution, approximate_inference=cached,
        actual_model_forward_calls=runtime.forward_calls,
        cfg_delta_trace=copy.deepcopy(runtime.cfg_delta_trace),
        cfg_delta_stats=copy.deepcopy(runtime.cfg_delta_stats),
        cfg_delta_policy=copy.deepcopy(POLICY) if cached else None)
    report.setdefault('sampler', {}).update(actual_model_forward_calls=runtime.forward_calls,
        model_forwards_per_step='2 exact / 1 approximate; see cfg_delta_trace' if cached else 2)
    report.setdefault('mathematical_changes', {}).update(
        text_cfg='exact:vu+5*(vc-vu);approximate:vc_current+4*delta_previous' if cached else
                 'v_negative + scale * (v_positive - v_negative)',
        guidance_delta_reuse=cached, cross_window_or_segment_cache=False,
        sampler_integration='unchanged original BF16 Euler and timestep labels')
    return report
