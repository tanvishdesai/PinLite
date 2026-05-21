"""
Quantized Gated-Attention (GAQ) research orchestrator.

This script provides a standardized experiment CLI interface:
  --exp_id --phase --seed --backend --attention_quant_policy
  --qat_epochs --calibration_profile --eval_profile

It also:
  - appends rows to research/experiment_registry.csv
  - updates research/agent_state.json
  - updates research/quant_gated_attention_main.md
  - prints one parseable summary line with CSV fields
  - prints exactly 3 prioritized Kaggle commands every cycle
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from gaq_attention_refactor import run_block_parity
except Exception:
    run_block_parity = None


PHASES = ("P0", "P1", "P2", "P3", "P4", "P5")
PHASE_NEXT = {"P0": "P1", "P1": "P2", "P2": "P3", "P3": "P4", "P4": "P5", "P5": "P5"}

ACC_DROP_LIMIT = 0.01
EPS_DROP_LIMIT = 0.05
SPEEDUP_MIN = 1.3

CSV_FIELDS = [
    "timestamp",
    "exp_id",
    "phase",
    "hypothesis",
    "code_ref",
    "seed",
    "dataset",
    "quant_backend",
    "attention_quant_policy",
    "qat_epochs",
    "calibration_profile",
    "acc",
    "f1",
    "auc",
    "eps",
    "spearman",
    "iou",
    "latency_cpu_ms",
    "size_mb",
    "speedup_vs_teacher",
    "pass_acc",
    "pass_eps",
    "pass_speed",
    "overall_pass",
    "status",
    "next_action",
]

FLOAT_FIELDS = {
    "acc",
    "f1",
    "auc",
    "eps",
    "spearman",
    "iou",
    "latency_cpu_ms",
    "size_mb",
    "speedup_vs_teacher",
}

INT_FIELDS = {"seed", "qat_epochs"}
BOOL_FIELDS = {"pass_acc", "pass_eps", "pass_speed", "overall_pass"}
COMPLETED_STATUSES = {"completed", "done", "success"}

POLICY_ORDER = [
    "fp32_reference",
    "hybrid_int8_qkv_out_gate_ffn_softmaxln_fp32",
    "hybrid_int8_qkv_out_gate_ffn_softmax_fp16_ln_fp32",
    "full_int8_attention_path",
]


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_optional_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    v = value.strip().lower()
    if v in {"1", "true", "yes", "y"}:
        return True
    if v in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean string: {value}")


def parse_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, float):
        return value
    text = str(value).strip()
    if text == "":
        return None
    return float(text)


def parse_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text == "":
        return None
    return int(text)


def read_json(path: Path, default_value: Dict[str, Any]) -> Dict[str, Any]:
    if not path.exists():
        return default_value
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def ensure_registry(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()


def load_registry(path: Path) -> List[Dict[str, Any]]:
    ensure_registry(path)
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for item in reader:
            row: Dict[str, Any] = {}
            for key in CSV_FIELDS:
                value = item.get(key, "")
                if key in FLOAT_FIELDS:
                    row[key] = parse_float(value)
                elif key in INT_FIELDS:
                    row[key] = parse_int(value)
                elif key in BOOL_FIELDS:
                    row[key] = parse_optional_bool(value) if value != "" else None
                else:
                    row[key] = value
            rows.append(row)
    return rows


def append_registry_row(path: Path, row: Dict[str, Any]) -> None:
    ensure_registry(path)
    write_row: Dict[str, Any] = {}
    for key in CSV_FIELDS:
        value = row.get(key)
        if value is None:
            write_row[key] = ""
        elif isinstance(value, bool):
            write_row[key] = "true" if value else "false"
        else:
            write_row[key] = str(value)
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writerow(write_row)


def default_state() -> Dict[str, Any]:
    return {
        "current_phase": "P0",
        "active_hypothesis": "Full PinPoint can be quantized end-to-end, including gated attention, while meeting accuracy, EPS, and CPU speed constraints.",
        "best_exp_id": "",
        "best_metrics": {},
        "failure_mode": "NOT_STARTED",
        "run_queue_top3": initial_queue(),
        "blocked_items": [],
        "decision_log": [
            {
                "timestamp": utc_now_iso(),
                "event": "initialized",
                "detail": "State was auto-created by gaq_experiment.py",
            }
        ],
        "baseline_metrics": {},
        "phase_gates": {phase: False for phase in PHASES},
    }


def is_completed(row: Dict[str, Any]) -> bool:
    status = (row.get("status") or "").strip().lower()
    return status in COMPLETED_STATUSES


def completed_rows(rows: List[Dict[str, Any]], phase: Optional[str] = None) -> List[Dict[str, Any]]:
    out = [r for r in rows if is_completed(r)]
    if phase is not None:
        out = [r for r in out if r.get("phase") == phase]
    return out


def median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(statistics.median(values))


def has_metrics(row: Dict[str, Any], keys: Tuple[str, ...]) -> bool:
    for key in keys:
        if row.get(key) is None:
            return False
    return True


def policy_decrease_aggressiveness(policy: str) -> str:
    if policy not in POLICY_ORDER:
        return "hybrid_int8_qkv_out_gate_ffn_softmaxln_fp32"
    idx = POLICY_ORDER.index(policy)
    return POLICY_ORDER[max(0, idx - 1)]


def next_exp_id(phase: str, rows: List[Dict[str, Any]]) -> str:
    prefix = f"GAQ-{phase}-"
    best = 0
    for row in rows:
        exp_id = (row.get("exp_id") or "").strip()
        if not exp_id.startswith(prefix):
            continue
        suffix = exp_id.replace(prefix, "")
        if suffix.isdigit():
            best = max(best, int(suffix))
    return f"{prefix}{best + 1:03d}"


def build_command(
    exp_id: str,
    phase: str,
    seed: int,
    backend: str,
    attention_quant_policy: str,
    qat_epochs: int,
    calibration_profile: str,
    eval_profile: str,
    status: str = "planned",
    extra: str = "",
) -> str:
    base = (
        f"python research/gaq_experiment.py --exp_id {exp_id} --phase {phase} --seed {seed} "
        f"--backend {backend} --attention_quant_policy {attention_quant_policy} --qat_epochs {qat_epochs} "
        f"--calibration_profile {calibration_profile} --eval_profile {eval_profile} --status {status}"
    )
    if extra:
        base = f"{base} {extra}".strip()
    return base


def initial_queue() -> List[Dict[str, Any]]:
    return [
        {
            "rank": 1,
            "label": "nearest-to-pass",
            "rationale": "Build FP32 teacher baseline with reproducible CPU metrics and EPS.",
            "command": build_command(
                exp_id="GAQ-P0-001",
                phase="P0",
                seed=11,
                backend="fp32_cpu",
                attention_quant_policy="fp32_reference",
                qat_epochs=0,
                calibration_profile="none",
                eval_profile="cpu_edge_v1",
            ),
        },
        {
            "rank": 2,
            "label": "highest-EPS-recovery-potential",
            "rationale": "Validate explicit Q/K/V parity before any quantization of gated attention.",
            "command": build_command(
                exp_id="GAQ-P1-001",
                phase="P1",
                seed=11,
                backend="fp32_cpu",
                attention_quant_policy="explicit_qkv_refactor",
                qat_epochs=0,
                calibration_profile="none",
                eval_profile="parity_v1",
                status="completed",
                extra="--run_parity",
            ),
        },
        {
            "rank": 3,
            "label": "highest-CPU-speedup-potential",
            "rationale": "Run first hybrid PTQ policy with softmax/LN high precision.",
            "command": build_command(
                exp_id="GAQ-P2-001",
                phase="P2",
                seed=11,
                backend="fbgemm",
                attention_quant_policy="hybrid_int8_qkv_out_gate_ffn_softmaxln_fp32",
                qat_epochs=0,
                calibration_profile="lavdf_200",
                eval_profile="cpu_edge_v1",
            ),
        },
    ]


def compute_passes(
    row: Dict[str, Any],
    baseline: Dict[str, Any],
    phase: str,
    parity_pass: Optional[bool],
) -> Tuple[bool, bool, bool, bool]:
    if not is_completed(row):
        return False, False, False, False

    if phase == "P0":
        return True, True, True, True

    if phase == "P1":
        gate_ok = bool(parity_pass)
        return gate_ok, gate_ok, gate_ok, gate_ok

    base_acc = baseline.get("acc")
    base_eps = baseline.get("eps")
    base_latency = baseline.get("latency_cpu_ms")

    acc = row.get("acc")
    eps = row.get("eps")
    speedup = row.get("speedup_vs_teacher")
    latency = row.get("latency_cpu_ms")

    if speedup is None and (latency is not None) and (base_latency is not None) and latency > 0:
        speedup = base_latency / latency
        row["speedup_vs_teacher"] = speedup

    pass_acc = bool((acc is not None) and (base_acc is not None) and (acc >= (base_acc - ACC_DROP_LIMIT)))
    pass_eps = bool((eps is not None) and (base_eps is not None) and (eps >= (base_eps - EPS_DROP_LIMIT)))
    pass_speed = bool((speedup is not None) and (speedup >= SPEEDUP_MIN))
    overall_pass = bool(pass_acc and pass_eps and pass_speed)
    return pass_acc, pass_eps, pass_speed, overall_pass


def evaluate_p0_gate(rows: List[Dict[str, Any]]) -> Tuple[bool, Dict[str, float]]:
    p0 = [r for r in completed_rows(rows, "P0") if has_metrics(r, ("acc", "eps", "latency_cpu_ms"))]
    if len(p0) < 3:
        return False, {}

    p0 = p0[-3:]
    accs = [r["acc"] for r in p0]
    epss = [r["eps"] for r in p0]
    lats = [r["latency_cpu_ms"] for r in p0]

    std_acc = statistics.pstdev(accs)
    std_eps = statistics.pstdev(epss)
    mean_lat = statistics.mean(lats)
    std_lat = statistics.pstdev(lats)
    cv_lat = (std_lat / mean_lat) if mean_lat > 0 else 999.0

    stable = (std_acc <= 0.003) and (std_eps <= 0.01) and (cv_lat <= 0.10)
    baseline = {
        "acc": float(statistics.median(accs)),
        "eps": float(statistics.median(epss)),
        "latency_cpu_ms": float(statistics.median(lats)),
    }
    return stable, baseline


def evaluate_p3_gate(rows: List[Dict[str, Any]], baseline: Dict[str, Any]) -> bool:
    p3 = [r for r in completed_rows(rows, "P3") if has_metrics(r, ("acc", "eps"))]
    if len(p3) < 3:
        return False
    p3 = p3[-3:]
    med_acc = median([r["acc"] for r in p3])
    med_eps = median([r["eps"] for r in p3])
    med_speed = median([r["speedup_vs_teacher"] for r in p3 if r["speedup_vs_teacher"] is not None])
    if med_acc is None or med_eps is None or med_speed is None:
        return False
    base_acc = baseline.get("acc")
    base_eps = baseline.get("eps")
    if base_acc is None or base_eps is None:
        return False
    return (med_acc >= base_acc - ACC_DROP_LIMIT) and (med_eps >= base_eps - EPS_DROP_LIMIT) and (med_speed >= SPEEDUP_MIN)


def evaluate_p4_gate(rows: List[Dict[str, Any]], state: Dict[str, Any]) -> bool:
    p4 = [r for r in completed_rows(rows, "P4") if r.get("eps") is not None]
    if not p4:
        return False
    if any(bool(r.get("overall_pass")) for r in p4):
        return True
    best_p3_eps = None
    best_metrics = state.get("best_metrics") or {}
    if (state.get("current_phase") in {"P4", "P5"}) and (best_metrics.get("phase") == "P3"):
        best_p3_eps = best_metrics.get("eps")
    if best_p3_eps is None:
        p3 = [r for r in completed_rows(rows, "P3") if r.get("eps") is not None and bool(r.get("overall_pass"))]
        if p3:
            best_p3_eps = max(float(r["eps"]) for r in p3)
    if best_p3_eps is None or len(p4) < 2:
        return False
    last_two = p4[-2:]
    return all(float(r["eps"]) < (float(best_p3_eps) - EPS_DROP_LIMIT) for r in last_two)


def evaluate_p5_gate(rows: List[Dict[str, Any]]) -> bool:
    p5 = completed_rows(rows, "P5")
    if len(p5) < 4:
        return False
    seed_runs = [r for r in p5 if r.get("seed") is not None]
    cross_runs = [r for r in p5 if "cross" in (r.get("dataset") or "").lower()]
    return len(seed_runs) >= 3 and len(cross_runs) >= 1


def evaluate_phase_gate(
    phase: str,
    rows: List[Dict[str, Any]],
    state: Dict[str, Any],
    parity_pass: Optional[bool],
) -> Tuple[bool, Dict[str, Any]]:
    payload: Dict[str, Any] = {}
    if phase == "P0":
        ok, baseline = evaluate_p0_gate(rows)
        if ok:
            payload["baseline_metrics"] = baseline
        return ok, payload
    if phase == "P1":
        return bool(parity_pass), payload
    if phase == "P2":
        ok = any(bool(r.get("overall_pass")) for r in completed_rows(rows, "P2"))
        return ok, payload
    if phase == "P3":
        ok = evaluate_p3_gate(rows, state.get("baseline_metrics") or {})
        return ok, payload
    if phase == "P4":
        ok = evaluate_p4_gate(rows, state)
        return ok, payload
    if phase == "P5":
        ok = evaluate_p5_gate(rows)
        return ok, payload
    return False, payload


def infer_failure_mode(row: Dict[str, Any], phase: str) -> str:
    if not is_completed(row):
        return "WAITING_FOR_RESULTS"
    if phase == "P1" and not bool(row.get("overall_pass")):
        return "PARITY_MISMATCH"
    if bool(row.get("overall_pass")):
        return "NONE"
    if not bool(row.get("pass_acc")):
        return "ACC_REGRESSION"
    if not bool(row.get("pass_eps")):
        return "EPS_DRIFT"
    if not bool(row.get("pass_speed")):
        return "INSUFFICIENT_SPEEDUP"
    return "PHASE_GATE_PENDING"


def infer_next_action(row: Dict[str, Any], phase: str) -> str:
    if not is_completed(row):
        return "Collect metrics for this run, then re-evaluate gates."
    if phase == "P1":
        if bool(row.get("overall_pass")):
            return "Advance to P2 hybrid PTQ with softmax/LN kept in high precision."
        return "Fix explicit Q/K/V parity until max abs error and mean abs error are within threshold."
    if bool(row.get("overall_pass")):
        return "Run robustness replication with a new seed; do not change architecture."
    if not bool(row.get("pass_acc")):
        return "Reduce quant aggressiveness by one notch or increase QAT epochs."
    if not bool(row.get("pass_eps")):
        return "Keep softmax/LN in high precision and tune attention calibration before increasing coverage."
    if not bool(row.get("pass_speed")):
        return "Change backend/runtime settings first before changing model math."
    return "Continue current phase and collect additional evidence."


def select_best_candidate(rows: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    candidates = [r for r in rows if is_completed(r) and bool(r.get("overall_pass")) and r.get("phase") in {"P2", "P3", "P4", "P5"}]
    if not candidates:
        return "", {}

    def score(r: Dict[str, Any]) -> float:
        acc = float(r.get("acc") or 0.0)
        eps = float(r.get("eps") or 0.0)
        speed = float(r.get("speedup_vs_teacher") or 0.0)
        return (0.4 * speed) + (0.35 * acc) + (0.25 * eps)

    best = max(candidates, key=score)
    metrics = {
        "phase": best.get("phase"),
        "acc": best.get("acc"),
        "f1": best.get("f1"),
        "auc": best.get("auc"),
        "eps": best.get("eps"),
        "spearman": best.get("spearman"),
        "iou": best.get("iou"),
        "latency_cpu_ms": best.get("latency_cpu_ms"),
        "size_mb": best.get("size_mb"),
        "speedup_vs_teacher": best.get("speedup_vs_teacher"),
    }
    return str(best.get("exp_id") or ""), metrics


def queue_item(rank: int, label: str, rationale: str, command: str) -> Dict[str, Any]:
    return {"rank": rank, "label": label, "rationale": rationale, "command": command}


def generate_queue(latest: Dict[str, Any], rows: List[Dict[str, Any]], state: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not latest:
        return initial_queue()

    phase = state.get("current_phase", latest.get("phase", "P0"))
    seed = int(latest.get("seed") or 11)
    backend = str(latest.get("quant_backend") or "fbgemm")
    policy = str(latest.get("attention_quant_policy") or "hybrid_int8_qkv_out_gate_ffn_softmaxln_fp32")
    qat = int(latest.get("qat_epochs") or 0)
    calib = str(latest.get("calibration_profile") or "lavdf_200")
    eval_profile = str(latest.get("eval_profile") or "cpu_edge_v1")

    if phase == "P0":
        return initial_queue()

    if bool(latest.get("overall_pass")):
        exp1 = next_exp_id(phase, rows)
        exp1_num = int(exp1.split("-")[-1])
        exp2 = f"GAQ-{phase}-{exp1_num + 1:03d}"
        exp3 = f"GAQ-{phase}-{exp1_num + 2:03d}"
        return [
            queue_item(
                1,
                "nearest-to-pass",
                "Robustness replicate with a new seed and same settings.",
                build_command(exp1, phase, seed + 1, backend, policy, qat, calib, eval_profile, status="planned"),
            ),
            queue_item(
                2,
                "highest-EPS-recovery-potential",
                "Second robustness replicate to support 3-seed claims.",
                build_command(exp2, phase, seed + 2, backend, policy, qat, calib, eval_profile, status="planned"),
            ),
            queue_item(
                3,
                "highest-CPU-speedup-potential",
                "Cross-dataset snapshot once same-config robustness is underway.",
                build_command(exp3, phase, seed + 3, backend, policy, qat, calib, "cross_dataset_snapshot_v1", status="planned"),
            ),
        ]

    exp_a = next_exp_id(phase, rows)
    exp_a_num = int(exp_a.split("-")[-1])
    exp_b = f"GAQ-{phase}-{exp_a_num + 1:03d}"
    exp_c = f"GAQ-{phase}-{exp_a_num + 2:03d}"

    pass_acc = bool(latest.get("pass_acc"))
    pass_eps = bool(latest.get("pass_eps"))
    pass_speed = bool(latest.get("pass_speed"))

    if not pass_acc:
        less_aggressive = policy_decrease_aggressiveness(policy)
        return [
            queue_item(
                1,
                "nearest-to-pass",
                "Reduce quant aggressiveness and increase QAT epochs to recover accuracy.",
                build_command(exp_a, phase, seed + 1, backend, less_aggressive, qat + 2, calib, eval_profile, status="planned"),
            ),
            queue_item(
                2,
                "highest-EPS-recovery-potential",
                "Hold softmax/LN high precision with tighter calibration profile.",
                build_command(exp_b, phase, seed + 2, backend, "hybrid_int8_qkv_out_gate_ffn_softmaxln_fp32", qat + 1, "lavdf_500_attention_heavy", eval_profile, status="planned"),
            ),
            queue_item(
                3,
                "highest-CPU-speedup-potential",
                "Try backend/runtime-only tuning without changing model math.",
                build_command(exp_c, phase, seed + 3, "x86_onnx_int8", policy, qat, calib, "cpu_edge_speed_v2", status="planned"),
            ),
        ]

    if not pass_eps:
        return [
            queue_item(
                1,
                "nearest-to-pass",
                "Keep softmax/LN in high precision and tune calibration for attention maps.",
                build_command(exp_a, phase, seed + 1, backend, "hybrid_int8_qkv_out_gate_ffn_softmaxln_fp32", qat + 1, "lavdf_500_attention_heavy", eval_profile, status="planned"),
            ),
            queue_item(
                2,
                "highest-EPS-recovery-potential",
                "Reduce attention quantization aggressiveness one notch.",
                build_command(exp_b, phase, seed + 2, backend, policy_decrease_aggressiveness(policy), qat + 1, calib, eval_profile, status="planned"),
            ),
            queue_item(
                3,
                "highest-CPU-speedup-potential",
                "Probe speed with runtime backend changes while keeping attention-safe policy.",
                build_command(exp_c, phase, seed + 3, "x86_onnx_int8", "hybrid_int8_qkv_out_gate_ffn_softmaxln_fp32", qat, calib, "cpu_edge_speed_v2", status="planned"),
            ),
        ]

    if not pass_speed:
        return [
            queue_item(
                1,
                "nearest-to-pass",
                "Backend/runtime tuning first before model math changes.",
                build_command(exp_a, phase, seed + 1, "x86_onnx_int8", policy, qat, calib, "cpu_edge_speed_v2", status="planned"),
            ),
            queue_item(
                2,
                "highest-EPS-recovery-potential",
                "Keep current model math and re-check EPS under tuned runtime.",
                build_command(exp_b, phase, seed + 2, "x86_onnx_int8", policy, qat, calib, "cpu_edge_eps_v2", status="planned"),
            ),
            queue_item(
                3,
                "highest-CPU-speedup-potential",
                "Try hardware-specific backend profile with same quant policy.",
                build_command(exp_c, phase, seed + 3, "arm_qnnpack_int8", policy, qat, calib, "cpu_edge_speed_v3", status="planned"),
            ),
        ]

    return initial_queue()


def replace_block(text: str, start_marker: str, end_marker: str, block_content: str) -> str:
    if start_marker not in text or end_marker not in text:
        addition = f"\n{start_marker}\n{block_content}\n{end_marker}\n"
        return text + addition
    start_idx = text.index(start_marker) + len(start_marker)
    end_idx = text.index(end_marker)
    return text[:start_idx] + "\n" + block_content + "\n" + text[end_idx:]


def update_progress_doc(path: Path, state: Dict[str, Any], latest: Dict[str, Any]) -> None:
    if path.exists():
        text = path.read_text(encoding="utf-8")
    else:
        text = (
            "# Quantized Gated-Attention Main\n\n"
            "## Main Hypothesis\n\n"
            "## Current Best Candidate\n<!-- AUTO_CURRENT_BEST_START -->\n\n<!-- AUTO_CURRENT_BEST_END -->\n\n"
            "## Validated/Rejected Hypotheses\n\n"
            "## Failure Analysis\n<!-- AUTO_FAILURE_START -->\n\n<!-- AUTO_FAILURE_END -->\n\n"
            "## Next 3 Runs\n<!-- AUTO_NEXT3_START -->\n\n<!-- AUTO_NEXT3_END -->\n\n"
            "## What to run in Kaggle now\n<!-- AUTO_KAGGLE_START -->\n\n<!-- AUTO_KAGGLE_END -->\n"
        )

    best_id = state.get("best_exp_id") or "None"
    best = state.get("best_metrics") or {}
    best_block = (
        f"- Best experiment: `{best_id}`\n"
        f"- Phase: `{best.get('phase', 'N/A')}`\n"
        f"- Accuracy: `{best.get('acc', 'N/A')}`\n"
        f"- EPS: `{best.get('eps', 'N/A')}`\n"
        f"- CPU speedup: `{best.get('speedup_vs_teacher', 'N/A')}`"
    )

    failure_block = (
        f"- Current phase: `{state.get('current_phase', 'N/A')}`\n"
        f"- Failure mode: `{state.get('failure_mode', 'NONE')}`\n"
        f"- Latest experiment: `{latest.get('exp_id', 'N/A')}`\n"
        f"- Next action: `{latest.get('next_action', 'N/A')}`"
    )

    queue = state.get("run_queue_top3") or []
    next3_lines: List[str] = []
    kaggle_lines: List[str] = []
    for item in queue[:3]:
        rank = item.get("rank", "?")
        label = item.get("label", "candidate")
        rationale = item.get("rationale", "")
        command = item.get("command", "")
        next3_lines.append(f"{rank}. `{label}` - {rationale}")
        kaggle_lines.append(f"{rank}. `{command}`")

    next3_block = "\n".join(next3_lines) if next3_lines else "No queue available."
    kaggle_block = "\n".join(kaggle_lines) if kaggle_lines else "No commands available."

    text = replace_block(text, "<!-- AUTO_CURRENT_BEST_START -->", "<!-- AUTO_CURRENT_BEST_END -->", best_block)
    text = replace_block(text, "<!-- AUTO_FAILURE_START -->", "<!-- AUTO_FAILURE_END -->", failure_block)
    text = replace_block(text, "<!-- AUTO_NEXT3_START -->", "<!-- AUTO_NEXT3_END -->", next3_block)
    text = replace_block(text, "<!-- AUTO_KAGGLE_START -->", "<!-- AUTO_KAGGLE_END -->", kaggle_block)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GAQ experiment orchestrator and logger")
    parser.add_argument("--exp_id", required=True)
    parser.add_argument("--phase", required=True, choices=PHASES)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--attention_quant_policy", required=True)
    parser.add_argument("--qat_epochs", required=True, type=int)
    parser.add_argument("--calibration_profile", required=True)
    parser.add_argument("--eval_profile", required=True)

    parser.add_argument("--hypothesis", default="")
    parser.add_argument("--code_ref", default="research/gaq_experiment.py")
    parser.add_argument("--dataset", default="LAV-DF")
    parser.add_argument("--status", default="completed")
    parser.add_argument("--next_action", default="")

    parser.add_argument("--acc", type=float, default=None)
    parser.add_argument("--f1", type=float, default=None)
    parser.add_argument("--auc", type=float, default=None)
    parser.add_argument("--eps", type=float, default=None)
    parser.add_argument("--spearman", type=float, default=None)
    parser.add_argument("--iou", type=float, default=None)
    parser.add_argument("--latency_cpu_ms", type=float, default=None)
    parser.add_argument("--size_mb", type=float, default=None)
    parser.add_argument("--speedup_vs_teacher", type=float, default=None)

    parser.add_argument("--run_parity", action="store_true")
    parser.add_argument("--parity_trials", type=int, default=8)
    parser.add_argument("--parity_pass", default=None)

    parser.add_argument("--registry_path", default="research/experiment_registry.csv")
    parser.add_argument("--state_path", default="research/agent_state.json")
    parser.add_argument("--progress_path", default="research/quant_gated_attention_main.md")
    return parser


def main() -> None:
    args = make_parser().parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    registry_path = (repo_root / args.registry_path).resolve()
    state_path = (repo_root / args.state_path).resolve()
    progress_path = (repo_root / args.progress_path).resolve()

    ensure_registry(registry_path)
    state = read_json(state_path, default_state())
    rows = load_registry(registry_path)

    parity_result: Dict[str, Any] = {}
    parity_pass_override = parse_optional_bool(args.parity_pass)

    if args.run_parity:
        if run_block_parity is None:
            raise RuntimeError("Parity utility unavailable. Ensure research/gaq_attention_refactor.py is importable.")
        parity_result = run_block_parity(seed=args.seed, trials=args.parity_trials)
        if parity_pass_override is None:
            parity_pass_override = bool(parity_result.get("parity_pass"))

    row: Dict[str, Any] = {
        "timestamp": utc_now_iso(),
        "exp_id": args.exp_id,
        "phase": args.phase,
        "hypothesis": args.hypothesis,
        "code_ref": args.code_ref,
        "seed": args.seed,
        "dataset": args.dataset,
        "quant_backend": args.backend,
        "attention_quant_policy": args.attention_quant_policy,
        "qat_epochs": args.qat_epochs,
        "calibration_profile": args.calibration_profile,
        "acc": args.acc,
        "f1": args.f1,
        "auc": args.auc,
        "eps": args.eps,
        "spearman": args.spearman,
        "iou": args.iou,
        "latency_cpu_ms": args.latency_cpu_ms,
        "size_mb": args.size_mb,
        "speedup_vs_teacher": args.speedup_vs_teacher,
        "pass_acc": False,
        "pass_eps": False,
        "pass_speed": False,
        "overall_pass": False,
        "status": args.status,
        "next_action": "",
        "eval_profile": args.eval_profile,
    }

    pass_acc, pass_eps, pass_speed, overall_pass = compute_passes(
        row=row,
        baseline=(state.get("baseline_metrics") or {}),
        phase=args.phase,
        parity_pass=parity_pass_override,
    )
    row["pass_acc"] = pass_acc
    row["pass_eps"] = pass_eps
    row["pass_speed"] = pass_speed
    row["overall_pass"] = overall_pass
    row["next_action"] = args.next_action.strip() if args.next_action.strip() else infer_next_action(row, args.phase)

    append_registry_row(registry_path, row)
    rows = load_registry(registry_path)

    gate_pass, gate_payload = evaluate_phase_gate(
        phase=args.phase,
        rows=rows,
        state=state,
        parity_pass=parity_pass_override,
    )
    state.setdefault("phase_gates", {phase: False for phase in PHASES})
    state["phase_gates"][args.phase] = bool(gate_pass)

    if "baseline_metrics" in gate_payload:
        state["baseline_metrics"] = gate_payload["baseline_metrics"]

    if gate_pass:
        state["current_phase"] = PHASE_NEXT.get(args.phase, args.phase)
    else:
        state["current_phase"] = args.phase

    best_exp_id, best_metrics = select_best_candidate(rows)
    state["best_exp_id"] = best_exp_id
    state["best_metrics"] = best_metrics
    state["active_hypothesis"] = args.hypothesis or state.get("active_hypothesis", "")
    state["failure_mode"] = infer_failure_mode(row, args.phase)

    queue = generate_queue(latest=row, rows=rows, state=state)
    state["run_queue_top3"] = queue[:3]

    decision = {
        "timestamp": row["timestamp"],
        "event": "run_logged",
        "exp_id": args.exp_id,
        "phase": args.phase,
        "overall_pass": bool(row["overall_pass"]),
        "phase_gate_pass": bool(gate_pass),
        "failure_mode": state["failure_mode"],
    }
    if parity_result:
        decision["parity"] = parity_result
    state.setdefault("decision_log", []).append(decision)

    write_json(state_path, state)
    update_progress_doc(progress_path, state, row)

    summary_payload = {key: row.get(key) for key in CSV_FIELDS}
    print("GAQ_SUMMARY " + json.dumps(summary_payload, sort_keys=True))

    for item in queue[:3]:
        rank = int(item.get("rank", 0))
        command = str(item.get("command", ""))
        print(f"KAGGLE_CMD_{rank}={command}")

    if parity_result:
        print("PARITY_RESULT " + json.dumps(parity_result, sort_keys=True))


if __name__ == "__main__":
    main()
