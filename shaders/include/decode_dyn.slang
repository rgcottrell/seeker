// Per-forward dynamic parameters. Mirrors `src/inference/decode_dyn.rs`
// — keep the layout in sync. See that module's doc comment for the
// motivation (single-submit-per-token decode replay).

struct DecodeDyn {
    uint  kv_len;            // 0
    uint  k_num;             // 4
    uint  blocks_per_split;  // 8
    uint  rope_d_offset;     // 12
    float uniform_rng;       // 16
    uint  penalty_count;     // 20
    uint  _pad0;             // 24
    uint  _pad1;             // 28
};
