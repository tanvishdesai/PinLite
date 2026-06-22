# PIN-Lite v2 — GAQ Runbook (Kaggle)

This is the exact sequence to reproduce the GAQ results and build Table 1. Everything
runs on Kaggle. **The only file whose contents you must edit is `gaq_config.py`.**

---

## 0. One-time setup in a Kaggle notebook

The GAQ scripts import the teacher as `PinPoint` and the student as `Distill`. On
Kaggle, recreate those modules with `%%writefile`, then write the GAQ files.

```python
# Cell 1 — teacher module  (paste the FULL contents of PinPoint-main.py)
%%writefile PinPoint.py
# <contents of PinPoint-main.py>
```
```python
# Cell 2 — student module  (paste the FULL contents of Distill-student.py)
%%writefile Distill.py
# <contents of Distill-student.py>
```

Then add a Kaggle **Utility Script** (or `%%writefile`) for each of:
`gaq_config.py`, `gaq_core.py`, `gaq_selftest.py`, `gaq_p0_baseline.py`,
`gaq_p1_parity.py`, `gaq_ptq.py`, `gaq_qat.py`, `gaq_onnx_latency.py`,
`gaq_crossdataset.py`, `gaq_build_table.py`.

> If you instead keep the repo files as-is (hyphenated `PinPoint-main.py` /
> `Distill-student.py` next to the GAQ files), `gaq_config.py` will load them
> automatically — no renaming needed.

Install (ONNX latency only):
```bash
pip install onnx onnxruntime
```

---

## 1. EDIT THESE PATHS — `gaq_config.py`

Open `gaq_config.py` and set, under **section 1**:

| Variable | What it points to |
|---|---|
| `TEACHER_CKPT` | `best_pinpoint_model_antisocial.pth` (full FP32 teacher) |
| `STUDENT_CKPT` | `best_pinpoint_LITE_model.pth` (distilled student — the model we quantize) |
| `LAVDF_DATA_DIRECTORY` | LAV-DF preprocessed data dir |
| `LAVDF_METADATA_PATH` | LAV-DF `unified_metadata.json` |
| `FAVC_DATA_DIRECTORY` | FakeAVCeleb preprocessed data dir (cross-dataset only) |
| `FAVC_METADATA_PATH` | FakeAVCeleb metadata json (cross-dataset only) |
| `OUTPUT_DIR` | where CSVs/checkpoints are written (default `/kaggle/working`) |

`TARGET_CPU_NAME` should name the CPU you benchmark on (honest "named hardware").
Optionally tune `EPS_SAMPLES`, `FAITHFULNESS_SAMPLES`, `CALIBRATION_SAMPLES`, etc.

> These paths are the in-repo defaults; adjust the dataset slug to match your
> Kaggle inputs. Nothing else in the codebase needs editing.

---

## 2. Run order

| # | Command | HW | Produces |
|---|---|---|---|
| 1 | `python gaq_selftest.py` | CPU | sanity (no data needed) |
| 2 | `python gaq_p0_baseline.py` | GPU (T4/P100) | `gaq_results.csv` rows: Teacher FP32, Distilled student; `gaq_teacher_latency.txt` |
| 3 | `python gaq_p1_parity.py` | CPU/GPU | parity gate (must pass) |
| 4 | `python gaq_ptq.py` | GPU + CPU | rows: Naive INT8 strawman, GAQ-INT8 (PTQ); `gaq_int8_ptq.pth` |
| 5 | `python gaq_qat.py` | GPU (P100) | row: GAQ-QAT; `gaq_int8_qat.pth` |
| 6 | `python gaq_onnx_latency.py` | CPU | `gaq_latency.csv` (named-hardware latency) |
| 7 | `python gaq_crossdataset.py` | GPU | `gaq_crossdataset.csv` (needs FAVC paths) |
| 8 | `python gaq_build_table.py` | CPU | `gaq_table1.md` (final table + success bar) |

Steps 2 → 5 must run in order (each appends to `gaq_results.csv` and step 2 writes
the teacher-latency reference used for speedup). Steps 6–7 are independent. Step 8
reads whatever CSVs exist.

The story Table 1 tells: **naive INT8 destroys the attention map (EPS + faithfulness
collapse); GAQ-INT8 keeps both; GAQ-QAT recovers any residual loss** — all while
hitting ≥1.3× real CPU speedup on named hardware.

---

## 3. Optional — a *real* pruning result

Unstructured pruning (the old `Prunning.py`) is dropped: it zeroed weights without
removing them, so it produced no size change. If a reviewer wants a pruning result,
do **structured** channel pruning of the MobileNetV3 backbone (the unpruned ~55% of
params) so size/params actually drop, then fine-tune with KD. This is intentionally
left out of the headline because GAQ is the contribution and pruning is no longer
load-bearing.

---

## 4. Files you'll get in `OUTPUT_DIR`

- `gaq_results.csv` — master rows (acc/AUC/EPS/Spearman/IoU/deletion/insertion/faithfulness/latency/size/params/speedup)
- `gaq_latency.csv` — backend latency (PyTorch FP32/INT8, ONNX FP32/INT8)
- `gaq_crossdataset.csv` — LAV-DF→FakeAVCeleb FP32 vs GAQ
- `gaq_table1.md` — assembled Table 1 + success-bar verdict
- `gaq_int8_ptq.pth`, `gaq_int8_qat.pth` — calibrated/QAT GAQ checkpoints
- `gaq_student_fp32.onnx`, `gaq_student_int8.onnx` — exported models
