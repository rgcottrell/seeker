// Per-forward dynamic parameters. Mirrors `src/inference/decode_dyn.rs`
// — keep the layout in sync. See that module's doc comment for the
// motivation (single-submit-per-token decode replay).

struct DecodeDyn {
    uint  kv_len;             // 0
    uint  k_num;              // 4
    uint  blocks_per_split;   // 8
    uint  rope_d_offset;      // 12
    float uniform_rng;        // 16
    uint  penalty_count;      // 20
    uint  v_cache_d_offset;   // 24
    // Per-sequence KV-cache slab index for batched/continuous decode. The
    // flash-attn kernel reads its K/V from `slot * nb13` / `slot * nb23`,
    // letting a batch gather arbitrary (non-contiguous) BatchKvCache slabs —
    // so prefix-reuse can park a conversation in any slab and still join the
    // batch. Single-sequence decode leaves it 0 (slab 0). // 28
    uint  slot;
};
