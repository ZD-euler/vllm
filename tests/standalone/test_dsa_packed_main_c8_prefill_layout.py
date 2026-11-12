# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU geometry checks without importing the vLLM runtime."""

import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def api():
    path = Path(__file__).resolve().parents[2] / "vllm/v1/core/dsa_shared_pool.py"
    return SimpleNamespace(**runpy.run_path(str(path)))


@pytest.fixture
def packed(api):
    return api.dsa_packed_main_c8_prefill_layout(
        3, block_size=128, latent_dims=(512, 64), indexer_dim=128
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
    assert pool.layout.slot_count == 13
    # Include the null bundle and its scale storage in the child slab budget.
    assert pool.layout.slot_count * pool.layout.allocation_bytes_per_bundle(
        1, 1
    ) == 13 * 2728960


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


def test_existing_bf16_factory_preserves_prefill_layout(api):
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
    )
    layout = api.dsa_shared_block_layout(main, li, 3, prefill_child=True)
    assert type(layout) is api.DSASharedBlockLayout
    assert layout.latent_page_size_bytes == 147456
    assert layout.bundle_page_size_bytes == 589824
    assert layout.scale_bytes_per_bundle == 9216
    assert layout.allocation_bytes_per_bundle(1, 1) == 599040
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
