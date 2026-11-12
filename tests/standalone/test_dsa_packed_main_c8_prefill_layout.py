# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU geometry checks without importing the vLLM runtime."""

import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.v1.core.test_prefill_mixed_c8_shared_pool import setup_case, utilities


@pytest.fixture(scope="module")
def api():
    path = Path(__file__).resolve().parents[2] / "vllm/v1/core/dsa_shared_pool.py"
    return SimpleNamespace(**runpy.run_path(str(path)))


@pytest.fixture(params=[0, 2**21], ids=["unaligned", "2mib-backing"])
def packed(api, request):
    return api.dsa_packed_main_c8_prefill_layout(
        3,
        block_size=128,
        latent_dims=(512, 64),
        indexer_dim=128,
        backing_alignment_bytes=request.param,
    )


def test_packed_geometry_and_padding_budget(api, packed):
    assert packed.latent_dim == 576
    assert packed.latent_page_size_bytes == 128 * (512 + 128 + 16)
    assert packed.indexer_page_size_bytes == 32768
    assert packed.bundle_page_size_bytes == 2686976
    assert packed.latent_blocks_per_bundle == 32
    assert packed.indexer_blocks_per_bundle == 82
    assert packed.scale_bytes_per_bundle == 41984
    assert packed.allocation_bytes_per_bundle(1, 1) == 2728960

    pool = api.PrefillLayerBundlePool(api.DSASharedBundleAllocator(packed), 4)
    assert isinstance(pool.layout, api.DSAPackedMainC8PrefillLayout)
    assert pool.layout.backing_alignment_bytes == packed.backing_alignment_bytes
    assert pool.layout.allocation_overhead_bytes(4, 2) == packed.backing_alignment_bytes
    assert pool.layout.slot_count == 13
    # Include the null bundle and its scale storage in the child slab budget.
    assert pool.layout.slot_count * pool.layout.allocation_bytes_per_bundle(
        1, 1
    ) == 13 * 2728960


def test_prefill_sizing_charges_alignment_once(api, packed, monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE", "true")
    monkeypatch.delenv("VLLM_ASCEND_LAYERWISE_PREFILL_NUM_BANKS", raising=False)
    ns = utilities(api, True)
    config, groups, _ = setup_case(api, True)
    # Exercise the existing consumer in isolation with an explicit prototype.
    ns["dsa_shared_block_layout"] = lambda *_args, **_kwargs: packed
    payload = 9 * 2728960  # Four banks for each owner, plus slot zero.
    # Both alignment cases start with the same budget and no block override.
    # With the reserve charged, only one complete parent fits beside slot zero.
    natural = ns["get_kv_cache_config_from_groups"](config, groups, payload)
    expected_parents, expected_slots = (
        (1, 5) if packed.backing_alignment_bytes else (2, 9)
    )
    assert natural.num_blocks == expected_parents
    assert len(natural.kv_cache_tensors) == 1
    assert natural.kv_cache_tensors[0].size == expected_slots * 2728960

    budget = payload + packed.backing_alignment_bytes
    assert ns["_max_memory_usage_bytes_from_groups"](config, groups) == budget
    result = ns["get_kv_cache_config_from_groups"](config, groups, budget)
    assert result.num_blocks == 2  # Four child slots per parent.
    assert len(result.kv_cache_tensors) == 1
    assert result.kv_cache_tensors[0].size == payload

    config.cache_config.num_gpu_blocks_override = 2
    with pytest.raises(ValueError, match="exceeds the HBM budget"):
        ns["get_kv_cache_config_from_groups"](config, groups, budget - 1)


def test_banked_allocations_share_capacity(api, packed, monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE", "true")
    monkeypatch.delenv("VLLM_ASCEND_LAYERWISE_PREFILL_NUM_BANKS", raising=False)
    parent = api.DSASharedBundleAllocator(packed)
    pool = api.PrefillLayerBundlePool(parent, 4)
    assert pool.bank_count == 4
    main = api.DSASharedBlockOwner.LATENT
    li = api.DSASharedBlockOwner.INDEXER
    assert pool.bundle_count_for_blocks(main, 32) == 4
    assert pool.bundle_count_for_blocks(main, 33) == 8
    assert pool.bundle_count_for_blocks(li, 82) == 4
    assert pool.bundle_count_for_blocks(li, 83) == 8

    main_banks = pool.allocate_banks(main, 1)
    li_banks = pool.allocate_banks(li, 1)
    assert main_banks == ((1,), (4,), (7,), (10,))
    assert li_banks == ((2,), (5,), (8,), (11,))
    assert pool.free_bundle_count == 4
    with pytest.raises(ValueError, match="Cannot get"):
        pool.allocate_banks(main, 2)
    assert pool.free_bundle_count == 4

    layout = pool.layout
    for owner, banks, pitch in ((main, main_banks, 83968), (li, li_banks, 32768)):
        for bank, (bundle_id,) in enumerate(banks):
            pages = layout.block_ids_for_bundle(owner, bundle_id)
            assert pool.physical_slot(bundle_id) == bank
            parent_id = pool.parent_bundle_id(bundle_id)
            assert pool.child_bundle_id(bank, parent_id) == bundle_id
            assert pages[0] * pitch == bundle_id * layout.bundle_page_size_bytes
            assert (pages[-1] + 1) * pitch == (
                bundle_id + 1
            ) * layout.bundle_page_size_bytes
            assert all(layout.bundle_id_for_block(owner, p) == bundle_id for p in pages)

    with pytest.raises(ValueError, match="is owned by"):
        pool.free(li, main_banks[0])
    for bank in main_banks:
        pool.free(main, bank)
    assert parent.free_bundle_count == 2  # The LI request still owns one parent.
    for bank in li_banks:
        pool.free(li, bank)
    assert parent.free_bundle_count == 3


def test_null_and_out_of_range_ids_are_rejected(api, packed):
    for owner in (api.DSASharedBlockOwner.LATENT, api.DSASharedBlockOwner.INDEXER):
        for bundle_id in (0, packed.slot_count):
            with pytest.raises(ValueError, match="invalid bundle id"):
                packed.block_ids_for_bundle(owner, bundle_id)
        last_null_block = packed.blocks_per_bundle(owner) - 1
        for block_id in (0, last_null_block, packed.row_count(owner)):
            with pytest.raises(ValueError, match="not a real DSA block"):
                packed.bundle_id_for_block(owner, block_id)


@pytest.mark.parametrize(
    "block_size,latent_dims,indexer_dim",
    [(64, (512, 64), 128), (128, (512, 32), 128), (128, (512, 64), 64)],
)
def test_unsupported_geometry_is_rejected(api, block_size, latent_dims, indexer_dim):
    with pytest.raises(ValueError, match="Unsupported packed main C8 prefill shape"):
        api.dsa_packed_main_c8_prefill_layout(
            3, block_size=block_size, latent_dims=latent_dims, indexer_dim=indexer_dim
        )


@pytest.mark.parametrize("alignment", [0, 2**21])
def test_existing_bf16_factory_preserves_prefill_layout(api, alignment):
    bf16 = SimpleNamespace(itemsize=2)
    int8 = SimpleNamespace(itemsize=1)
    main = SimpleNamespace(
        page_size_bytes=147456, sparse_head_dim=(512, 64), dtype=bf16
    )
    li = SimpleNamespace(
        block_size=128,
        sparse_head_dim=(128,),
        dtype=bf16,
        c8_k_cache_dtype=int8,
        indexer_scale_page_size_bytes=256,
        shared_pool_alignment_bytes=alignment,
    )
    layout = api.dsa_shared_block_layout(main, li, 3, prefill_child=True)
    assert type(layout) is api.DSASharedBlockLayout
    assert layout.latent_page_size_bytes == 147456
    assert layout.bundle_page_size_bytes == 589824
    assert layout.scale_bytes_per_bundle == 9216
    assert layout.allocation_bytes_per_bundle(1, 1) == 599040
    assert layout.backing_alignment_bytes == alignment
    assert layout.block_ids_for_bundle(api.DSASharedBlockOwner.LATENT, 1) == tuple(
        range(4, 8)
    )
    assert layout.block_ids_for_bundle(api.DSASharedBlockOwner.INDEXER, 1) == (
        *range(16, 32),
        66,
        67,
    )

    main.page_size_bytes = 83968
    main.dtype = int8
    with pytest.raises(ValueError, match="ratio does not match"):
        api.dsa_shared_block_layout(main, li, 3, prefill_child=True)


@pytest.mark.parametrize("alignment", [-1, 1.5, True])
def test_invalid_backing_alignment_is_rejected(api, alignment):
    with pytest.raises(ValueError, match="backing alignment must be"):
        api.dsa_packed_main_c8_prefill_layout(
            3,
            block_size=128,
            latent_dims=(512, 64),
            indexer_dim=128,
            backing_alignment_bytes=alignment,
        )
