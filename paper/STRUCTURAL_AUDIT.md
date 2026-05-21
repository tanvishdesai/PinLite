# Structural Audit - PIN-Lite Draft

**Draft audited:** `paper/PIN_Lite_NPL_Draft.md`
**Venue checklist:** `paper/VENUE_NOTES.md`
**Date:** 2026-05-14

## Present Elements

- [x] Abstract in journal style.
- [x] 4-6 keywords.
- [x] Numbered introduction contributions.
- [x] Methods comparison table via technical dump and related-work synthesis.
- [x] Main benchmark table.
- [x] Ablation table.
- [x] Computational cost table/columns.
- [x] Cross-dataset evaluation table.
- [x] EPS custom metric equation.
- [x] Distillation loss equations.
- [x] Concrete limitations and future work.
- [x] Declarations section.
- [x] Numeric references.

## Present But Needs Improvement

- [ ] Figure placeholders are present, but final figure files should be inserted and cited in order.
- [ ] Qualitative teacher/student attention maps are not yet generated in final paper form.
- [ ] Hardware/software details are incomplete.
- [ ] LAV-DF train/dev class distribution is incomplete.
- [ ] Raw FakeAVCeleb cross-dataset CSV was not found; the table currently relies on the rough PDF ledger.
- [ ] References need final Springer formatting and DOI verification.

## Missing Before Submission

- [ ] Exact GPU model, VRAM, PyTorch, CUDA, torchvision, librosa/scipy versions.
- [ ] Reproducible split counts and class distributions from the final metadata file.
- [ ] Final figure files at Springer-compatible resolution.
- [ ] Optional but recommended: a real confusion matrix for the selected final model.
- [ ] Human-authored final declarations and author-contribution statements.

## Next Revision Passes

1. Convert the Markdown draft to Springer Nature LaTeX or Word template.
2. Generate or insert final figures and make captions match Springer format.
3. Verify every metric against raw CSV/logs and remove the conflicting secondary pipeline numbers if they cannot be reconciled.
4. Tighten related work into fewer, more synthetic paragraphs after final citation verification.
5. Run a sentence-level pass after all missing experimental metadata is filled.

