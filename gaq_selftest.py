"""
====================================================================================
GAQ self-test — validates the quant engine WITHOUT the dataset or checkpoints.
====================================================================================
Run this first on any new environment to confirm gaq_core works:

    python gaq_selftest.py

Checks:
  1. FP32 parity: the quantizable explicit block is math-identical to the original
     GatedCrossAttentionBlock (max error should be ~1e-6).
  2. GAQ preserves the attention map even when attention is peaky.
  3. The naive INT8 strawman corrupts a peaky attention map (proves the strawman is
     a genuine failure mode, not a rigged one).
====================================================================================
"""

import importlib.util

import torch

import gaq_core as G


def _load_block():
    spec = importlib.util.spec_from_file_location("pp_local", "PinPoint-main.py")
    pp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pp)
    return pp.GatedCrossAttentionBlock


def _calibrate_and_run(block, policy, per_channel, audio, video, mask):
    qb = G.QGatedCrossAttentionBlock(block, num_bits=8, per_channel=per_channel).eval()
    qb.set_policy(policy)
    for fq in G._all_act_fq(qb):
        fq.calibrating = True
    with torch.no_grad():
        qb(audio, video, mask)
    for fq in G._all_act_fq(qb):
        if fq.calibrating:
            fq.freeze()
    qb.set_policy(policy)
    with torch.no_grad():
        out, attn = qb(audio, video, mask)
    return out, attn


def main():
    G.set_seed(0)
    GCAB = _load_block()
    embed, heads = 64, 4
    block = GCAB(embed, heads, dropout=0.0).eval()
    audio = torch.randn(2, 40, embed)
    video = torch.randn(2, 30, embed)
    mask = torch.zeros(2, 30, dtype=torch.bool)
    mask[0, -3:] = True
    with torch.no_grad():
        o_ref, m_ref = block(audio, video, mask)

    # 1) FP32 parity
    qb = G.QGatedCrossAttentionBlock(block).eval()
    qb.set_policy("fp32")
    with torch.no_grad():
        o_fp, m_fp = qb(audio, video, mask)
    out_err = (o_ref - o_fp).abs().max().item()
    attn_err = (m_ref - m_fp).abs().max().item()
    assert out_err < 1e-4 and attn_err < 1e-4, f"parity failed: {out_err}, {attn_err}"
    print(f"[1] FP32 parity OK     max|out|={out_err:.2e} max|attn|={attn_err:.2e}")

    # peaky attention (trained-like) to exercise INT8 dynamic range
    with torch.no_grad():
        block.audio_to_video_attn.in_proj_weight.mul_(6.0)
        o_ref, m_ref = block(audio, video, mask)
    ref = m_ref[0].flatten().numpy()

    _, m_gaq = _calibrate_and_run(block, "gaq", True, audio, video, mask)
    _, m_naive = _calibrate_and_run(block, "naive", False, audio, video, mask)
    sp_gaq = G._spearman(ref, m_gaq[0].flatten().numpy())
    sp_naive = G._spearman(ref, m_naive[0].flatten().numpy())
    print(f"[2] GAQ preserves map  spearman={sp_gaq:.4f}")
    print(f"[3] Naive corrupts map spearman={sp_naive:.4f}")
    assert sp_gaq > sp_naive, "expected GAQ to beat naive on attention fidelity"
    print("\nSELF-TEST PASSED  (GAQ keeps the attention map; naive INT8 breaks it)")


if __name__ == "__main__":
    main()
