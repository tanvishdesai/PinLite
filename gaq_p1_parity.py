"""
====================================================================================
GAQ P1 — Explicit Q/K/V refactor parity test (MUST PASS before any quantization)
====================================================================================
The whole GAQ method rests on decomposing nn.MultiheadAttention into explicit
Q/K/V/out Linear ops. Before we quantize anything we must prove that decomposition
is numerically identical to the original gated cross-attention.

This script checks parity at the *model* level: the quantizable explicit student
(policy='fp32', i.e. fake-quant OFF) must produce the same classification logits
AND the same attention maps as the original distilled student.

Run:
    python gaq_p1_parity.py
====================================================================================
"""

import torch

import gaq_config as C
import gaq_core as G


def main():
    C.banner("GAQ P1 — EXPLICIT Q/K/V REFACTOR PARITY")
    G.set_seed(C.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n[1/3] Loading distilled student ...")
    student, s_cfg = G.load_student(device=device)
    student.eval()

    print("\n[2/3] Building quantizable explicit student (policy=fp32) ...")
    qstudent = G.build_quantizable_student(student, num_bits=8, per_channel=True)
    G.set_model_policy(qstudent, "fp32")
    qstudent.to(device).eval()

    print("\n[3/3] Comparing outputs on real LAV-DF batches ...")
    loaders = G.make_loaders(s_cfg, splits=("test",), batch_size=4)
    test_loader = loaders["test"]

    max_logit_err = 0.0
    max_attn_err = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in test_loader:
            if batch is None:
                continue
            v = batch["video"].to(device)
            a = batch["audio"].to(device)
            l_ref, _, m_ref = student(v, a)
            l_new, _, m_new = qstudent(v, a)
            max_logit_err = max(max_logit_err, (l_ref - l_new).abs().max().item())
            if m_ref is not None and m_new is not None:
                k = min(m_ref.shape[-1], m_new.shape[-1])
                t = min(m_ref.shape[-2], m_new.shape[-2])
                max_attn_err = max(
                    max_attn_err,
                    (m_ref[:, :t, :k] - m_new[:, :t, :k]).abs().max().item())
            n_batches += 1
            if n_batches >= 10:
                break

    atol = 1e-4
    passed = (max_logit_err <= atol) and (max_attn_err <= atol)
    print(f"\n  batches checked    : {n_batches}")
    print(f"  max |logit error|  : {max_logit_err:.2e}")
    print(f"  max |attn  error|  : {max_attn_err:.2e}")
    print(f"  tolerance          : {atol:.0e}")
    print(f"\n  PARITY {'PASSED ✅' if passed else 'FAILED ❌'}")

    # Optional: also run the original module-level parity harness if present.
    try:
        import importlib.util
        import os
        ref = os.path.join("research", "gaq_attention_refactor.py")
        if os.path.exists(ref):
            spec = importlib.util.spec_from_file_location("gaq_refactor", ref)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            res = mod.run_block_parity(seed=C.SEED, trials=8)
            print(f"\n  [module-level parity] {res}")
    except Exception as e:
        print(f"  [module-level parity skipped] {e}")

    if not passed:
        raise SystemExit("P1 parity failed — fix the explicit refactor before P2.")


if __name__ == "__main__":
    main()
