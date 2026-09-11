"""CPU-only small tensors: sampling changes without altering cache/model contracts."""

from collections import Counter
from itertools import islice
from types import SimpleNamespace

import pytest
import torch

from training.data import action_dataset as legacy
from training.data.action_resampled import (
    INDEX_POLICY,
    SAMPLING_NAMESPACE,
    ResampledActionDataset,
    build_resampled_action_teacher_dataloader,
    sampling_contract,
)


@pytest.fixture
def fake_cache(monkeypatch):
    calls = []

    def initialize(self, index_path, *, seed=42, timestep_shift=5.0, **kwargs):
        calls.append({"index_path": index_path, "seed": seed, **kwargs})
        self.seed = seed
        self.timestep_shift = timestep_shift
        self.episodes = [
            {"episode_id": f"train-{i:04d}", "num_windows": 8} for i in range(780)
        ]
        self.samples_per_episode = 8

    def window(self, episode, window_index):
        clean = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2, 1) / 12
        clean += window_index
        return {"clean_latents": clean, "actions": torch.full((8, 8), float(window_index)),
                "prompt_embeds": torch.ones(4, 8), "y": torch.ones(1)}

    monkeypatch.setattr(legacy.PrecomputedActionDataset, "__init__", initialize)
    monkeypatch.setattr(legacy.PrecomputedActionDataset, "_cached_window", window)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return calls


def _dataset(num_samples=8320):
    return ResampledActionDataset("unused", num_samples=num_samples, seed=42)


def _configs(steps=1040):
    return (
        SimpleNamespace(manifest_path="/data/manifests/train.jsonl", num_workers=0,
                        precomputed_latents=True, precomputed_text_embeddings=True,
                        prompt_cache_path="/data/features/static.pt"),
        SimpleNamespace(max_steps=steps, gradient_accumulation_steps=8,
                        micro_batch_size=1, seed=42),
    )


def test_contract_has_fixed_namespace_and_exact_virtual_budget():
    _, training = _configs(260)
    contract = sampling_contract(training)
    assert contract["namespace"] == SAMPLING_NAMESPACE == "action_resampled_absolute_v1"
    assert contract["index_policy"] == INDEX_POLICY
    assert contract["data_seed"] == legacy._seed64(42, SAMPLING_NAMESPACE)
    assert contract["training_seed"] == 42
    assert contract["virtual_samples"] == contract["micro_batches"] == 2080
    assert contract["last_absolute_sample_index"] == 2079


def test_same_index_is_exactly_reproducible_and_does_not_consume_global_rng(fake_cache):
    dataset = _dataset()
    rng = torch.get_rng_state().clone()
    first = dataset[6240]
    for key, value in first.items():
        assert torch.equal(value, dataset[6240][key])
        assert value.device.type == "cpu"
    assert torch.equal(rng, torch.get_rng_state())
    assert set(first) == {"noisy_latents", "target_flow", "timesteps", "actions", "prompt_embeds", "y"}


def test_new_namespace_does_not_replay_legacy_diffusion_seed(fake_cache):
    dataset = _dataset()
    for index in (0, 20, 6240, 8319):
        assert dataset.sampling_spec(index)["diffusion_seed"] != legacy._seed64(42, "diffusion", index)


def test_next_physical_pass_has_fresh_noise_and_timestep_draw(fake_cache):
    dataset = _dataset()
    first, next_pass = dataset[0], dataset[6240]
    assert dataset.sampling_spec(0)["episode_id"] == dataset.sampling_spec(6240)["episode_id"]
    assert dataset.sampling_spec(6240)["window_epoch"] == 1
    # Use target+clean to recover noise, allowing a changed physical window.
    noises = []
    for index, batch in ((0, first), (6240, next_pass)):
        spec = dataset.sampling_spec(index)
        clean = dataset._cached_window(dataset.episodes[spec["episode_slot"]], spec["window_index"])["clean_latents"]
        noises.append(batch["target_flow"][1:] + clean[1:])
    assert not torch.equal(noises[0], noises[1])
    # A specific deterministic pair differs; global uniqueness is NOT required.
    assert not torch.equal(first["timesteps"], next_pass["timesteps"])


def test_episode_balance_and_no_replacement_cover_all_cached_windows(fake_cache):
    dataset = _dataset(12480)
    prefix = [dataset.sampling_spec(i) for i in range(2080)]
    counts = Counter(row["episode_id"] for row in prefix)
    assert Counter(counts.values()) == {3: 520, 2: 260}
    assert len({(row["episode_id"], row["window_index"]) for row in prefix}) == 2080
    for slot in range(780):
        for epoch in (0, 1):
            windows = [dataset.sampling_spec(slot + 780 * (epoch * 8 + j))["window_index"] for j in range(8)]
            assert sorted(windows) == list(range(8))


def test_unequal_window_counts_remain_episode_balanced(fake_cache):
    dataset = _dataset(18)
    dataset.episodes = [{"episode_id": "a", "num_windows": 1},
                        {"episode_id": "b", "num_windows": 3}]
    specs = [dataset.sampling_spec(i) for i in range(18)]
    assert Counter(row["episode_id"] for row in specs) == {"a": 9, "b": 9}
    assert all(row["window_index"] == 0 for row in specs if row["episode_id"] == "a")
    for offset in range(0, 9, 3):
        assert sorted(specs[1 + 2 * (offset + j)]["window_index"] for j in range(3)) == [0, 1, 2]


def test_flow_equation_and_clean_initial_condition_unchanged(fake_cache):
    dataset = _dataset()
    spec = dataset.sampling_spec(123)
    clean = dataset._cached_window(dataset.episodes[spec["episode_slot"]], spec["window_index"])["clean_latents"]
    generator = torch.Generator(device="cpu").manual_seed(spec["diffusion_seed"])
    noise = torch.randn(clean.shape, generator=generator)
    schedule_index = int(torch.randint(0, 1000, (1,), generator=generator))
    linear = 1 - schedule_index / 1000
    sigma = 5 * linear / (1 + 4 * linear)
    batch = dataset[123]
    assert torch.equal(batch["noisy_latents"][0], clean[0])
    assert torch.count_nonzero(batch["target_flow"][0]) == 0
    assert batch["timesteps"][0] == 0
    assert torch.equal(batch["noisy_latents"][1:], ((1 - sigma) * clean + sigma * noise)[1:])
    assert torch.equal(batch["target_flow"][1:], (noise - clean)[1:])
    assert torch.allclose(batch["timesteps"][1:], torch.full((2,), sigma * 1000))


def test_factory_budget_is_virtual_not_physical_and_keeps_cache_contract(fake_cache):
    config, training = _configs()
    loader = build_resampled_action_teacher_dataloader(config=config, training=training)
    assert len(loader) == len(loader.dataset) == 8320
    assert len(loader.dataset.episodes) * loader.dataset.samples_per_episode == 6240
    assert fake_cache[-1]["split"] == "train"
    assert fake_cache[-1]["manifest_path"] == config.manifest_path
    assert fake_cache[-1]["prompt_cache_path"] == config.prompt_cache_path
    assert fake_cache[-1]["seed"] == 42
    assert loader.drop_last and not loader.persistent_workers


def test_trainer_resume_position_after_6240_and_extended_run_does_not_wrap(fake_cache, monkeypatch):
    from train_action_teacher import _iterator_at_micro_batch

    config, training = _configs(1060)
    loader = build_resampled_action_teacher_dataloader(config=config, training=training)
    # Fast metadata-only batches exercise the real trainer iterator boundary.
    monkeypatch.setattr(ResampledActionDataset, "__getitem__",
                        lambda self, index: self.sampling_spec(index)["absolute_sample_index"])
    _, resumed = _iterator_at_micro_batch(loader, 8320)
    assert [int(batch.item()) for batch in islice(resumed, 3)] == [8320, 8321, 8322]
    _, physical_boundary = _iterator_at_micro_batch(loader, 6240)
    assert int(next(physical_boundary).item()) == 6240


def test_run_extension_and_worker_access_order_preserve_samples(fake_cache):
    original, extended = _dataset(8320), _dataset(8480)
    for index in (8319, 0, 6240, 17):
        assert original.sampling_spec(index) == extended.sampling_spec(index)
        for key, tensor in original[index].items():
            assert torch.equal(tensor, extended[index][key])


def test_legacy_factory_and_bounded_repeat_contract_are_unchanged(fake_cache):
    config, training = _configs()
    old = legacy.build_action_teacher_dataloader(config=config, training=training)
    assert type(old.dataset) is legacy.PrecomputedActionDataset
    assert len(old) == 6240
    original = old.dataset[0]
    assert torch.equal(original["noisy_latents"], old.dataset[6240 % len(old)]["noisy_latents"])
    assert not hasattr(old.dataset, "data_seed")


def test_real_cache_static_binding_and_train_split_remain_enforced(tmp_path):
    from test_action_dataset import _static_prompt_fixture

    path, index, manifest, shard, _, receipt = _static_prompt_fixture(tmp_path)
    before = {file: legacy.sha256_file(file) for file in (path, index, manifest, shard)}
    dataset = ResampledActionDataset(index, num_samples=4, manifest_path=manifest,
                                     prompt_cache_path=path, seed=42)
    assert [row["episode_id"] for row in dataset.episodes] == ["train-a", "train-b"]
    assert dataset.prompt_cache_receipt == receipt
    first, repeated_window = dataset[0], dataset[2]
    assert torch.all(first["prompt_embeds"] == 2)
    assert torch.equal(first["actions"], repeated_window["actions"])
    assert not torch.equal(first["noisy_latents"], repeated_window["noisy_latents"])
    first["prompt_embeds"].fill_(99)
    assert torch.all(dataset[0]["prompt_embeds"] == 2)
    assert {file: legacy.sha256_file(file) for file in before} == before


@pytest.mark.parametrize("value", [0, -1, True, 1.25, "2080"])
def test_invalid_virtual_budget_rejected_before_cache_read(value, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid budget must not open cache")
    monkeypatch.setattr(legacy.PrecomputedActionDataset, "__init__", forbidden)
    with pytest.raises(ValueError, match="positive integer"):
        ResampledActionDataset("unused", num_samples=value)


def test_index_bounds(fake_cache):
    dataset = _dataset(10)
    assert dataset.sampling_spec(-1) == dataset.sampling_spec(9)
    for index in (-11, 10):
        with pytest.raises(IndexError):
            dataset[index]
    for index in (True, 1.1):
        with pytest.raises(TypeError):
            dataset[index]


@pytest.mark.parametrize("field", ["precomputed_latents", "precomputed_text_embeddings"])
def test_factory_rejects_raw_feature_fallback(field):
    config, training = _configs()
    setattr(config, field, False)
    with pytest.raises(ValueError, match=field):
        build_resampled_action_teacher_dataloader(config=config, training=training)
