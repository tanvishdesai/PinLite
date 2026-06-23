# PIN-Lite v2 — GAQ Runbook (P5 frontier session)

Focused runbook for the **new** code only. Assumes you already ran P0→QAT in a
previous session and kept their outputs (`gaq_results.csv`, `gaq_crossdataset.csv`).

---

## 0. Files to put in this session

Re-create the modules and scripts. **Two files changed since last time — make sure
you upload the NEW versions**, not the old ones:

| File | Status | Notes |
|---|---|---|
| `PinPoint.py` | unchanged | `%%writefile` from `PinPoint-main.py` |
| `Distill.py` | unchanged | `%%writefile` from `Distill-student.py` |
| `gaq_config.py` | unchanged | paths already set |
| `gaq_core.py` | **UPDATED** | `compute_eps` now returns per-sample EPS + tail stats — must re-upload |
| `gaq_p5_frontier.py` | **NEW** | the frontier sweep |
| `gaq_plot_frontier.py` | **NEW** | the figure |
| `gaq_onnx_latency.py` | **UPDATED** | hardened ONNX error handling |
| `gaq_build_table.py` | unchanged | reads existing CSVs |

## 0b. Re-seed OUTPUT_DIR with your previous results (only needed for step 4)

`gaq_build_table.py` reads CSVs from `OUTPUT_DIR` (`/kaggle/working`). In a fresh
session those are gone. Upload your previous-run files into `/kaggle/working` first,
otherwise the final table will be missing the teacher/student/PTQ/QAT/cross-dataset
rows:

- `gaq_results.csv`   ← **required** (Teacher/Student/Naive/GAQ-PTQ/GAQ-QAT rows)
- `gaq_crossdataset.csv` ← optional (adds the cross-dataset block)

Steps 1–3 below do **not** need these — they are self-contained.

```python
import shutil, os
os.makedirs("/kaggle/working", exist_ok=True)
shutil.copy("/kaggle/input/<your-prior-results>/gaq_results.csv", "/kaggle/working/")
shutil.copy("/kaggle/input/<your-prior-results>/gaq_crossdataset.csv", "/kaggle/working/")
```

---

## 1. Run order

| # | Command | HW | Produces | Needs |
|---|---|---|---|---|
| 1 | `python gaq_p5_frontier.py` | GPU | `gaq_frontier.csv`, `gaq_frontier_eps.csv` + printed verdict | checkpoints + LAV-DF (already configured) |
| 2 | `python gaq_plot_frontier.py` | CPU | `gaq_frontier.png` | `gaq_frontier.csv` from step 1 |
| 3 | `pip install onnx onnxruntime` then `python gaq_onnx_latency.py` | CPU | `gaq_latency.csv` | student checkpoint |
| 4 | `python gaq_build_table.py` | CPU | `gaq_table1.md` | the CSVs from step 0b + step 3 |

Steps 1, 3 are independent and self-contained. Step 2 needs step 1. Step 4 reads
whatever CSVs are present in `OUTPUT_DIR`.

```bash
python gaq_p5_frontier.py
python gaq_plot_frontier.py
pip install onnx onnxruntime
python gaq_onnx_latency.py
python gaq_build_table.py
```

---

## 2. What to read off the run

**Step 1 is the decision point.** It prints a verdict block:

```
FRONTIER VERDICT (EPS gap = GAQ - naive at each bit-width)
  INT8:  GAQ eps=...  naive eps=...  gap=+...
  INT6:  ...
  INT4:  ...
  INT3:  ...
```

- **Gap widens as bits drop** → the GAQ contribution is demonstrated (ICASSP story).
- **Gap stays ~0 even at INT3** → INT8 quant of this block is benign; honest claim
  shrinks to "1.4× smaller, iso-accuracy, iso-reasoning."

**Step 3:** report the ONNX-RT INT8 speedup **vs the FP32 student** (printed inline).
If it's still ~1×, the win is size, not latency — state that honestly.

> Note: `gaq_build_table.py` does **not** include the frontier — that lives in
> `gaq_frontier.csv` / `gaq_frontier.png` and the printed verdict. Paste the verdict
> block back and we'll decide framing/venue.

---

## 3. If step 1 is too slow

It evaluates 8 models (4 bit-widths × 2 policies). To trim:
- lower `FAITHFULNESS_SAMPLES` to ~60 in `gaq_config.py`, or
- set `FRONTIER_BITS = [8, 4, 3]` at the top of `gaq_p5_frontier.py`.
