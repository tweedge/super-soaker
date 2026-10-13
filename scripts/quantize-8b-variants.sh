#!/bin/bash
# scripts/quantize-8b-variants.sh <variant> — build custom LFM2.5-8B-A1B quants
# from BF16. This is the experimental record of the 2026-10 optimization
# campaign (notes/04-decode-optimization-research.md); the shipped profile
# stays on Unsloth UD-IQ2_M because every custom variant was quality-gated
# against bench/ha_battery.json and none beat it on tool accuracy.
#
# Usage:
#   scripts/quantize-8b-variants.sh <variant>
#   variants: min-bytes draft-fid q2k-experts prot-q2k prot-q3k prot-q40
#             prot-q2k-plus prot-iq2m-fast tq-fast iq2xxs-fastconv iq2m-draftfid
# Requires: LiquidAI LFM2.5-8B-A1B-BF16.gguf + unsloth imatrix in
# /var/lib/super-soaker/models (bootstrap fetches none of these by default).
#
# Variants (all keep attn_v/attn_output/embeddings high — quality levers):
#   min-bytes   experts q2_0/q2_0/q4_0, shortconv q2_0, embd q4_k
#               (validates "speed ~ 1/bytes" at fixed 24 GB/s streaming)
#   draft-fid   min-bytes BUT the 5 DSpark target layers (2,6,10,14,18)
#               get q4_k experts (drafter reads those hidden states)
#   q2k-experts experts q2_k/q2_k/q2_k (Orin MMQ crossover native), rest q4_k
#
# Uses unsloth's 1.5M-token imatrix when present.
set -euo pipefail
VARIANT="${1:?usage: quantize_custom.sh min-bytes|draft-fid|q2k-experts|prot-q2k|prot-q40|prot-q3k|prot-q2k-plus|prot-iq2m-fast|tq-fast|iq2xxs-fastconv|iq2m-draftfid}"
M=/var/lib/super-soaker/models
Q=/opt/super-soaker/build/llama.cpp/build/bin/llama-quantize
SRC="$M/LFM2.5-8B-A1B-BF16.gguf"
IMAT="$M/imatrix_unsloth.gguf_file"
OUT="$M/LFM2.5-8B-A1B-CUSTOM-${VARIANT}.gguf"
DRAFT_LAYERS='blk\.(2|6|10|14|18)\.'

[ -x "$Q" ] || { echo "llama-quantize missing"; exit 1; }
[ -f "$SRC" ] || { echo "BF16 source missing"; exit 1; }
if [ -f "$OUT" ]; then echo "have $OUT"; sha256sum "$OUT"; exit 0; fi

ARGS=()
if [ -f "$IMAT" ]; then ARGS+=(--imatrix "$IMAT"); else echo "WARN: no imatrix"; fi

case "$VARIANT" in
  min-bytes)
    ARGS+=(--pure
      --tensor-type 'ffn_gate_exps=q2_0'
      --tensor-type 'ffn_up_exps=q2_0'
      --tensor-type 'ffn_down_exps=q4_0'
      --tensor-type 'shortconv=q2_0'
      --tensor-type 'attn_v=q4_k'
      --tensor-type 'attn_output=q4_k')
    ;;
  draft-fid)
    # NOTE: --tensor-type is first-match-wins in CLI order — specific
    # draft-layer patterns MUST precede the generic expert patterns.
    ARGS+=(--pure
      --tensor-type "${DRAFT_LAYERS}ffn_gate_exps=q4_k"
      --tensor-type "${DRAFT_LAYERS}ffn_up_exps=q4_k"
      --tensor-type "${DRAFT_LAYERS}ffn_down_exps=q5_k"
      --tensor-type 'ffn_gate_exps=q2_0'
      --tensor-type 'ffn_up_exps=q2_0'
      --tensor-type 'ffn_down_exps=q4_0'
      --tensor-type 'shortconv=q2_0'
      --tensor-type 'attn_v=q4_k'
      --tensor-type 'attn_output=q4_k')
    ;;
  q2k-experts)
    ARGS+=(--pure
      --tensor-type 'ffn_.*_exps=q2_k'
      --tensor-type 'shortconv=q2_k'
      --tensor-type 'attn_v=q4_k'
      --tensor-type 'attn_output=q4_k')
    ;;
  prot-q2k)
    # Unsloth UD-Q2_K_XL protection (Q5_K/Q6_K on embd/attn/dense) with
    # fast-streaming Q2_K on experts+shortconv only.
    ARGS+=(--pure
      --output-tensor-type q6_k --token-embedding-type q5_k
      --tensor-type 'ffn_gate_exps=q2_k'
      --tensor-type 'ffn_up_exps=q2_k'
      --tensor-type 'ffn_down_exps=q2_k'
      --tensor-type 'shortconv=q2_k'
      --tensor-type 'attn_v=q5_k'
      --tensor-type 'attn_output=q5_k'
      --tensor-type 'attn_q=q5_k'
      --tensor-type 'attn_k=q6_k'
      --tensor-type 'ffn_gate=q5_k'
      --tensor-type 'ffn_up=q5_k'
      --tensor-type 'ffn_down=q6_k')
    ;;
  prot-q40)
    # Same protection, Q4_0 experts (4.5bpw, streams like q2_0 at 37GB/s class).
    ARGS+=(--pure
      --output-tensor-type q6_k --token-embedding-type q5_k
      --tensor-type 'ffn_gate_exps=q4_0'
      --tensor-type 'ffn_up_exps=q4_0'
      --tensor-type 'ffn_down_exps=q4_0'
      --tensor-type 'shortconv=q4_0'
      --tensor-type 'attn_v=q5_k'
      --tensor-type 'attn_output=q5_k'
      --tensor-type 'attn_q=q5_k'
      --tensor-type 'attn_k=q6_k'
      --tensor-type 'ffn_gate=q5_k'
      --tensor-type 'ffn_up=q5_k'
      --tensor-type 'ffn_down=q6_k')
    ;;
  prot-q3k)
    ARGS+=(--pure
      --output-tensor-type q6_k --token-embedding-type q5_k
      --tensor-type 'ffn_gate_exps=q3_k'
      --tensor-type 'ffn_up_exps=q3_k'
      --tensor-type 'ffn_down_exps=q3_k'
      --tensor-type 'shortconv=q3_k'
      --tensor-type 'attn_v=q5_k'
      --tensor-type 'attn_output=q5_k'
      --tensor-type 'attn_q=q5_k'
      --tensor-type 'attn_k=q6_k'
      --tensor-type 'ffn_gate=q5_k'
      --tensor-type 'ffn_up=q5_k'
      --tensor-type 'ffn_down=q6_k')
    ;;
  prot-q2k-plus)
    # prot-q2k with quality-critical down projections at Q5_K (1/3 of expert
    # traffic only); gate/up stay Q2_K (fast stream).
    ARGS+=(--pure
      --output-tensor-type q6_k --token-embedding-type q5_k
      --tensor-type 'ffn_gate_exps=q2_k'
      --tensor-type 'ffn_up_exps=q2_k'
      --tensor-type 'ffn_down_exps=q5_k'
      --tensor-type 'shortconv=q4_k'
      --tensor-type 'attn_v=q5_k'
      --tensor-type 'attn_output=q5_k'
      --tensor-type 'attn_q=q5_k'
      --tensor-type 'attn_k=q6_k'
      --tensor-type 'ffn_gate=q5_k'
      --tensor-type 'ffn_up=q5_k'
      --tensor-type 'ffn_down=q6_k')
    ;;
  prot-iq2m-fast)
    # UD-IQ2_M fidelity recipe (its exact protection profile) but experts
    # gate/up at Q4_0 instead of IQ2_XXS (37GB/s stream vs 24), down at Q5_K.
    ARGS+=(--pure
      --output-tensor-type q6_k --token-embedding-type q5_k
      --tensor-type 'ffn_gate_exps=q4_0'
      --tensor-type 'ffn_up_exps=q4_0'
      --tensor-type 'ffn_down_exps=q5_k'
      --tensor-type 'shortconv=q4_k'
      --tensor-type 'attn_v=q5_k'
      --tensor-type 'attn_output=q5_k'
      --tensor-type 'attn_q=q5_k'
      --tensor-type 'attn_k=q6_k'
      --tensor-type 'ffn_gate=q5_k'
      --tensor-type 'ffn_up=q5_k'
      --tensor-type 'ffn_down=q6_k')
    ;;
  tq-fast)
    # TQ2_0 ternary experts (fastest 2-bit on Ampere per PR #11183) +
    # protected rest. Does ternary preserve tool decisions?
    ARGS+=(--pure
      --output-tensor-type q6_k --token-embedding-type q5_k
      --tensor-type 'ffn_gate_exps=tq2_0'
      --tensor-type 'ffn_up_exps=tq2_0'
      --tensor-type 'ffn_down_exps=tq2_0'
      --tensor-type 'shortconv=q4_0'
      --tensor-type 'attn_v=q5_k'
      --tensor-type 'attn_output=q5_k'
      --tensor-type 'attn_q=q5_k'
      --tensor-type 'attn_k=q6_k'
      --tensor-type 'ffn_gate=q5_k'
      --tensor-type 'ffn_up=q5_k'
      --tensor-type 'ffn_down=q6_k')
    ;;
  iq2xxs-fastconv)
    # UD-IQ2_M expert types exactly (IQ2_XXS gate/up, IQ3_XXS down = tool-safe)
    # but fast-streaming Q4_0 shortconv (14% of active traffic off the LUT path).
    ARGS+=(--pure
      --output-tensor-type q6_k --token-embedding-type q5_k
      --tensor-type 'ffn_gate_exps=iq2_xxs'
      --tensor-type 'ffn_up_exps=iq2_xxs'
      --tensor-type 'ffn_down_exps=iq3_xxs'
      --tensor-type 'shortconv=q4_0'
      --tensor-type 'attn_v=q5_k'
      --tensor-type 'attn_output=q5_k'
      --tensor-type 'attn_q=q5_k'
      --tensor-type 'attn_k=q6_k'
      --tensor-type 'ffn_gate=q5_k'
      --tensor-type 'ffn_up=q5_k'
      --tensor-type 'ffn_down=q6_k')
    ;;
  iq2m-draftfid)
    # UD-IQ2_M tool-safe expert types, BUT the 5 DSpark target layers
    # (2,6,10,14,18) get Q5_K/Q6_K experts: the drafter consumes these
    # hidden states — higher fidelity may raise novel-content acceptance.
    # shortconv Q4_0 (quality-neutral per iq2xxs-fastconv gate).
    ARGS+=(--pure
      --output-tensor-type q6_k --token-embedding-type q5_k
      --tensor-type 'blk\.(2|6|10|14|18)\.ffn_gate_exps=q5_k'
      --tensor-type 'blk\.(2|6|10|14|18)\.ffn_up_exps=q5_k'
      --tensor-type 'blk\.(2|6|10|14|18)\.ffn_down_exps=q6_k'
      --tensor-type 'ffn_gate_exps=iq2_xxs'
      --tensor-type 'ffn_up_exps=iq2_xxs'
      --tensor-type 'ffn_down_exps=iq3_xxs'
      --tensor-type 'shortconv=q4_0'
      --tensor-type 'attn_v=q5_k'
      --tensor-type 'attn_output=q5_k'
      --tensor-type 'attn_q=q5_k'
      --tensor-type 'attn_k=q6_k'
      --tensor-type 'ffn_gate=q5_k'
      --tensor-type 'ffn_up=q5_k'
      --tensor-type 'ffn_down=q6_k')
    ;;
  *) echo "unknown variant $VARIANT"; exit 1 ;;
esac

echo "=== quantizing $VARIANT -> $OUT ==="
"$Q" "${ARGS[@]}" "$SRC" "$OUT" Q4_K_S
echo "=== done ==="
ls -la "$OUT"
sha256sum "$OUT"
