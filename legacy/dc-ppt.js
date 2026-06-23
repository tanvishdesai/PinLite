const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.title = "Fifth DC Review – PIN-Lite";
pres.author = "Palak Parmar";

// ── PALETTE ────────────────────────────────────────────────────────────────
const C = {
  navy:    "1B3D6B",
  teal:    "005F56",
  white:   "FFFFFF",
  near:    "1A1A2E",
  lgray:   "EEF2F7",
  mgray:   "9CA3AF",
  accent:  "E8F0F8",   // very light blue for alternating rows
  gold:    "D4A017",
  warn:    "C0392B",
};

const FONT = "Calibri";

// ── HELPERS ────────────────────────────────────────────────────────────────
/** Header bar + PDEU/SOT text badges */
function addHeader(slide, title) {
  // top navy bar
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 10, h: 0.55,
    fill: { color: C.navy }, line: { color: C.navy }
  });
  // title text
  slide.addText(title, {
    x: 0.18, y: 0, w: 7.5, h: 0.55,
    fontFace: FONT, fontSize: 18, bold: true,
    color: C.white, valign: "middle", margin: 0
  });
  // PDEU badge
  slide.addText("PDEU", {
    x: 7.75, y: 0.05, w: 1.0, h: 0.45,
    fontFace: FONT, fontSize: 9, bold: true, color: C.white,
    align: "center", valign: "middle",
    fill: { color: "163560" }, line: { color: "163560" }
  });
  // SOT badge
  slide.addText("SOT", {
    x: 8.78, y: 0.05, w: 0.95, h: 0.45,
    fontFace: FONT, fontSize: 9, bold: true, color: C.white,
    align: "center", valign: "middle",
    fill: { color: C.teal }, line: { color: C.teal }
  });
}

/** Teal footer bar */
function addFooter(slide, slideNum, labelText) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 5.3, w: 10, h: 0.325,
    fill: { color: C.teal }, line: { color: C.teal }
  });
  const label = labelText || "PIN-Lite: Lightweight Multimodal Deepfake Detection with Explainability Preservation";
  slide.addText(label, {
    x: 0.15, y: 5.3, w: 8.5, h: 0.325,
    fontFace: FONT, fontSize: 8, color: C.white,
    valign: "middle", margin: 0
  });
  slide.addText(String(slideNum), {
    x: 9.1, y: 5.3, w: 0.7, h: 0.325,
    fontFace: FONT, fontSize: 9, bold: true,
    color: C.white, align: "center", valign: "middle", margin: 0
  });
}

/** Dark section-box header (for two-panel slides) */
function panelBox(slide, x, y, w, h, color, text) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h,
    fill: { color }, line: { color }
  });
  slide.addText(text, {
    x, y, w, h,
    fontFace: FONT, fontSize: 12, bold: true, color: C.white,
    align: "center", valign: "middle", margin: 4
  });
}

/** Light card box with left accent */
function accentCard(slide, x, y, w, h, accentColor, headerText, bodyLines, headerFontSize = 11, bodyFontSize = 9.5) {
  // card bg
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h,
    fill: { color: C.lgray }, line: { color: "D8E2EE", pt: 0.5 }
  });
  // left accent
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w: 0.06, h,
    fill: { color: accentColor }, line: { color: accentColor }
  });
  let ty = y + 0.06;
  if (headerText) {
    slide.addText(headerText, {
      x: x + 0.1, y: ty, w: w - 0.15, h: 0.22,
      fontFace: FONT, fontSize: headerFontSize, bold: true,
      color: accentColor, margin: 0
    });
    ty += 0.25;
  }
  if (bodyLines && bodyLines.length) {
    const remaining = (y + h) - ty - 0.06;
    slide.addText(
      bodyLines.map((l, i) => ({
        text: l,
        options: { breakLine: i < bodyLines.length - 1 }
      })),
      {
        x: x + 0.1, y: ty, w: w - 0.15, h: remaining,
        fontFace: FONT, fontSize: bodyFontSize, color: C.near,
        valign: "top", margin: 0
      }
    );
  }
}

/** Literature review table row */
function litRow(slide, x, y, w, h, ref, year, methods, journal, dataset, perf, gap, isAlt) {
  const bg = isAlt ? C.accent : C.white;
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: bg }, line: { color: "D0D8E8", pt: 0.4 } });
  const cells = [ref, year, methods, journal, dataset, perf, gap];
  const widths = [0.45, 0.38, 1.4, 1.3, 0.85, 0.72, 2.2];
  let cx = x;
  cells.forEach((cell, i) => {
    slide.addText(cell, {
      x: cx + 0.03, y: y + 0.03, w: widths[i] - 0.06, h: h - 0.06,
      fontFace: FONT, fontSize: 7.5, color: C.near,
      valign: "middle", margin: 0, wrap: true
    });
    cx += widths[i];
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 1 ── TITLE
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.navy };

  // top decorative bar
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 10, h: 0.08, fill: { color: C.teal }, line: { color: C.teal }
  });
  // bottom bar
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 5.545, w: 10, h: 0.08, fill: { color: C.teal }, line: { color: C.teal }
  });

  // institution badges row
  s.addText("PDEU", {
    x: 0.5, y: 0.18, w: 1.1, h: 0.45,
    fontFace: FONT, fontSize: 10, bold: true, color: C.white,
    align: "center", valign: "middle",
    fill: { color: "163560" }, line: { color: "2B5EA7" }
  });
  s.addText("SOT", {
    x: 1.7, y: 0.18, w: 1.1, h: 0.45,
    fontFace: FONT, fontSize: 10, bold: true, color: C.white,
    align: "center", valign: "middle",
    fill: { color: C.teal }, line: { color: C.teal }
  });
  s.addText("UGC Recognized", {
    x: 2.95, y: 0.28, w: 1.8, h: 0.25,
    fontFace: FONT, fontSize: 8, color: "A0BAD8", italic: true, margin: 0
  });

  // main title
  s.addText("Fifth Doctoral Committee Review Presentation", {
    x: 0.5, y: 1.0, w: 9, h: 0.5,
    fontFace: FONT, fontSize: 22, bold: true,
    color: "A0C4E8", align: "center", valign: "middle"
  });
  s.addText("Deepfake Detection : A Multimodal Approach\nUsing Video and Audio Data", {
    x: 0.5, y: 1.55, w: 9, h: 1.0,
    fontFace: FONT, fontSize: 30, bold: true,
    color: C.white, align: "center", valign: "middle"
  });

  // divider
  s.addShape(pres.shapes.RECTANGLE, {
    x: 2.5, y: 2.7, w: 5, h: 0.05,
    fill: { color: C.teal }, line: { color: C.teal }
  });

  // date
  s.addText("15th May, 2026  |  11:30 AM – 12:30 PM", {
    x: 0.5, y: 2.82, w: 9, h: 0.35,
    fontFace: FONT, fontSize: 13, color: "A0BAD8", align: "center"
  });

  // people boxes
  const people = [
    { label: "Ph.D. Research Scholar", name: "Palak Parmar", info: "En: 24RCP002\nCSE-SoT, PDEU" },
    { label: "Supervisor", name: "Dr. SantoshKumar Bharti", info: "Assistant Professor\nCSE-SoT, PDEU" },
    { label: "External Supervisor", name: "Dr. Chintan Bhatt", info: "Assistant Professor\nUniversity of Wollongong" },
  ];
  const bx = [0.35, 3.6, 6.85];
  people.forEach((p, i) => {
    s.addShape(pres.shapes.RECTANGLE, {
      x: bx[i], y: 3.32, w: 3.05, h: 1.6,
      fill: { color: "162B50" }, line: { color: "2B5EA7", pt: 0.7 }
    });
    s.addText(p.label, {
      x: bx[i] + 0.1, y: 3.38, w: 2.85, h: 0.28,
      fontFace: FONT, fontSize: 8.5, bold: true, color: "A0C4E8",
      align: "center", margin: 0
    });
    s.addText(p.name, {
      x: bx[i] + 0.1, y: 3.68, w: 2.85, h: 0.35,
      fontFace: FONT, fontSize: 10, bold: true, color: C.white,
      align: "center", margin: 0
    });
    s.addText(p.info, {
      x: bx[i] + 0.1, y: 4.05, w: 2.85, h: 0.78,
      fontFace: FONT, fontSize: 8.5, color: "C0D8F0",
      align: "center", margin: 0
    });
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 2 ── TABLE OF CONTENTS
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  addHeader(s, "Table of Contents");
  addFooter(s, "2 / 24");

  const items = [
    ["01", "Comments from 4th DC Review"],
    ["02", "PinPoint — Contribution 1 Recap"],
    ["03", "Limitations of PinPoint & Motivation"],
    ["04", "Problem Statement & Research Gap"],
    ["05", "Literature Review"],
    ["06", "Objectives"],
    ["07", "Dataset"],
    ["08", "PIN-Lite Framework Overview"],
    ["09", "Architectural Evolution: Teacher → Student"],
    ["10", "Phase 1 — Attention-Aware Knowledge Distillation"],
    ["11", "Phase 2 — Iterative Pruning"],
    ["12", "Phase 3 — Post-Training Quantization"],
    ["13", "Experimental Setup"],
    ["14", "Main Results"],
    ["15", "Comparative Analysis"],
    ["16", "Conclusion"],
    ["17", "Future Work"],
  ];

  const col1 = items.slice(0, 9);
  const col2 = items.slice(9);

  col1.forEach((it, i) => {
    const y = 0.72 + i * 0.49;
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.25, y, w: 0.45, h: 0.35,
      fill: { color: C.navy }, line: { color: C.navy }
    });
    s.addText(it[0], {
      x: 0.25, y, w: 0.45, h: 0.35,
      fontFace: FONT, fontSize: 10, bold: true, color: C.white,
      align: "center", valign: "middle", margin: 0
    });
    s.addText(it[1], {
      x: 0.78, y: y + 0.04, w: 3.9, h: 0.28,
      fontFace: FONT, fontSize: 10, color: C.near, margin: 0
    });
  });

  col2.forEach((it, i) => {
    const y = 0.72 + i * 0.49;
    s.addShape(pres.shapes.RECTANGLE, {
      x: 5.05, y, w: 0.45, h: 0.35,
      fill: { color: C.teal }, line: { color: C.teal }
    });
    s.addText(it[0], {
      x: 5.05, y, w: 0.45, h: 0.35,
      fontFace: FONT, fontSize: 10, bold: true, color: C.white,
      align: "center", valign: "middle", margin: 0
    });
    s.addText(it[1], {
      x: 5.58, y: y + 0.04, w: 4.2, h: 0.28,
      fontFace: FONT, fontSize: 10, color: C.near, margin: 0
    });
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 3 ── COMMENTS FROM 4TH DC REVIEW
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  addHeader(s, "Comments from 4th DC Review");
  addFooter(s, "3 / 24");

  // Two-column layout: feedback received | action taken
  panelBox(s, 0.2, 0.65, 4.55, 0.4, C.navy, "Feedback Received (December 2025)");
  panelBox(s, 4.9, 0.65, 4.9, 0.4, C.teal, "Actions Taken");

  const feedback = [
    ["Frame the Research Objectives", "Objectives lacked clear scope and methodology for each contribution."],
    ["Focus on Publication", "Papers needed to be publication-ready and submitted to indexed journals."],
    ["Implement Contribution 2", "The second proposed work (PIN-Lite) needed to be fully implemented."],
  ];
  const actions = [
    ["Objectives Restructured", "Objectives now tightly aligned with actual deliverables: PinPoint (Contrib. 1) and PIN-Lite (Contrib. 2)."],
    ["PinPoint Paper Drafted", "First paper (PinPoint) drafted for submission; currently under review."],
    ["PIN-Lite Fully Implemented", "Complete compression pipeline (KD → Pruning → Quantization) implemented and evaluated on LAV-DF."],
  ];

  feedback.forEach(([h, b], i) => {
    const y = 1.14 + i * 1.3;
    accentCard(s, 0.2, y, 4.55, 1.18, C.navy, h, [b], 10, 9);
  });
  actions.forEach(([h, b], i) => {
    const y = 1.14 + i * 1.3;
    accentCard(s, 4.9, y, 4.9, 1.18, C.teal, h, [b], 10, 9);
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 4 ── PINPOINT RECAP
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  addHeader(s, "Contribution 1 Recap — PinPoint Architecture");
  addFooter(s, "4 / 24");

  // Left column: description
  const leftItems = [
    ["Interpretable Multimodal System", "Designed to detect deepfakes and precisely localize spatial and temporal artifacts for forensic transparency."],
    ["Architecture", "Fuses a ResNet-18 (video encoder) and CNN-GRU (audio extractor) via a Gated Cross-Attention Transformer."],
    ["Key Mechanisms", "Gated Cross-Attention + Synchronization Loss to model fine-grained audiovisual dependencies and temporal offsets."],
    ["Explainability (XAI Suite)", "Pinpoints inconsistencies using Integrated Gradients, SHAP, LIME, Grad-CAM, TCAV, and Counterfactuals."],
    ["Validation", "97.47% accuracy on unified benchmark (LAV-DF + FakeAVCeleb); validated with quantitative XAI metrics."],
  ];

  leftItems.forEach(([h, b], i) => {
    accentCard(s, 0.2, 0.68 + i * 0.89, 4.7, 0.82, C.navy, h, [b], 9.5, 8.5);
  });

  // Right column: performance table + architecture placeholder
  panelBox(s, 5.1, 0.68, 4.7, 0.38, C.navy, "PinPoint Performance Metrics");

  const rows = [
    ["Accuracy", "97.47%"],
    ["F1-Score", "0.982"],
    ["Precision", "0.984"],
    ["Recall", "0.980"],
    ["AUC", "0.968"],
    ["Model Size", "57.32 MB  |  15.0M params"],
    ["Inference Latency", "98.62 ms / sample"],
  ];
  rows.forEach(([k, v], i) => {
    const bg = i % 2 === 0 ? C.accent : C.white;
    const y = 1.12 + i * 0.3;
    s.addShape(pres.shapes.RECTANGLE, {
      x: 5.1, y, w: 4.7, h: 0.3, fill: { color: bg }, line: { color: "D0D8E8", pt: 0.4 }
    });
    s.addText(k, {
      x: 5.18, y: y + 0.03, w: 2.3, h: 0.24,
      fontFace: FONT, fontSize: 9, bold: true, color: C.navy, margin: 0
    });
    s.addText(v, {
      x: 7.5, y: y + 0.03, w: 2.25, h: 0.24,
      fontFace: FONT, fontSize: 9, color: C.near, margin: 0
    });
  });

  // Architecture figure placeholder
  s.addShape(pres.shapes.RECTANGLE, {
    x: 5.1, y: 3.28, w: 4.7, h: 1.85,
    fill: { color: "F0F4FA" }, line: { color: "B0C4DE", pt: 0.8 }
  });
  s.addText("[FIGURE PLACEHOLDER]\nPinPoint Architecture Diagram\n(Fig. 3 from 4th DC Presentation)", {
    x: 5.1, y: 3.28, w: 4.7, h: 1.85,
    fontFace: FONT, fontSize: 9, color: C.mgray,
    align: "center", valign: "middle"
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 5 ── LIMITATIONS OF PINPOINT → MOTIVATION
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  addHeader(s, "Limitations of PinPoint — The Case for Compression");
  addFooter(s, "5 / 24");

  // 5 limitation cards in a 2+3 layout
  const lims = [
    { t: "High Model Complexity", b: "15.0M parameters; 3× stacked Gated Cross-Attention blocks with 8 heads each — prohibitively expensive for lightweight deployment.", c: C.warn },
    { t: "Large Memory Footprint", b: "57.32 MB on disk; 886 MB peak VRAM during inference — exceeds typical edge device budgets.", c: C.warn },
    { t: "Slow Inference", b: "98.62 ms per sample on GPU — insufficient for real-time detection pipelines on surveillance cameras or mobile applications.", c: C.navy },
    { t: "Resource-Intensive Deployment", b: "Requires high-end GPU infrastructure; not feasible for embedded systems, IoT devices, or browser-side inference.", c: C.navy },
    { t: "XAI Suite Overhead", b: "The comprehensive XAI suite (SHAP, IG, TCAV) adds significant computation per sample, making real-time explanations impractical.", c: C.teal },
  ];

  lims.slice(0, 2).forEach((l, i) => {
    accentCard(s, 0.2 + i * 4.85, 0.68, 4.65, 2.1, l.c, l.t, [l.b], 10.5, 9.5);
  });
  lims.slice(2).forEach((l, i) => {
    accentCard(s, 0.2 + i * 3.25, 2.88, 3.1, 2.1, l.c, l.t, [l.b], 10, 9);
  });

  // Central arrow / bridge text
  s.addShape(pres.shapes.RECTANGLE, {
    x: 6.35, y: 2.9, w: 3.4, h: 2.1,
    fill: { color: C.navy }, line: { color: C.navy }
  });
  s.addText([
    { text: "Therefore —\n", options: { bold: true, breakLine: false } },
    { text: "\nPIN-Lite was motivated by the need to preserve explainability while dramatically reducing model complexity and inference overhead.\n\nTarget: Edge-deployable real-time deepfake detection.", options: {} }
  ], {
    x: 6.43, y: 2.95, w: 3.22, h: 1.98,
    fontFace: FONT, fontSize: 10, color: C.white,
    valign: "middle", align: "center"
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 6 ── PROBLEM STATEMENT & RESEARCH GAP
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  addHeader(s, "Problem Statement & Research Gap");
  addFooter(s, "6 / 24");

  panelBox(s, 0.2, 0.65, 4.65, 0.38, C.navy, "Gap in the Literature");
  panelBox(s, 5.1, 0.65, 4.7, 0.38, C.teal, "How PIN-Lite Fills the Gap");

  const gaps = [
    ["Visual-Only Compression", "Prior compression baselines (Karathanasis, LightFakeDetect) target visual-only models. They completely miss crucial audio-visual synchronization artifacts that are central to multimodal forgery detection."],
    ["Spurious Features in KD", "Standard Knowledge Distillation causes students to learn spurious background features, silently degrading explanation consistency — a critical risk in forensic applications."],
    ["No Reasoning Fidelity Metric", "There is no formal, architecture-agnostic metric to verify whether a compressed model retains the original model's attention-map reasoning, not just its output accuracy."],
  ];
  const fills = [
    ["Multimodal Compression", "First framework to jointly compress audio + video deepfake detection models while addressing cross-modal synchronization artifacts."],
    ["Attention-Aware KD", "Three-component distillation loss (L_hard + L_soft + L_attn) explicitly transfers the teacher's audio-visual reasoning patterns to the student."],
    ["10.83× Compression", "Systematic Distill → Prune → Quantize pipeline achieving 57.32 MB → 5.29 MB with 97.37% → 98.22% accuracy gain."],
  ];

  gaps.forEach(([h, b], i) => {
    accentCard(s, 0.2, 1.1 + i * 1.35, 4.65, 1.27, C.warn, h, [b], 10, 8.5);
  });
  fills.forEach(([h, b], i) => {
    accentCard(s, 5.1, 1.1 + i * 1.35, 4.7, 1.27, C.teal, h, [b], 10, 8.5);
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 7 ── LITERATURE REVIEW I  (2025 entries — accuracy but no efficiency)
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  addHeader(s, "Literature Review — High-Performance Models (2025)");
  addFooter(s, "7 / 24");

  // Table headers
  const hdr = [
    { text: "Ref", w: 0.42 }, { text: "Year", w: 0.36 }, { text: "Methods", w: 1.38 },
    { text: "Venue", w: 1.28 }, { text: "Dataset", w: 0.83 }, { text: "Perf.", w: 0.7 }, { text: "Research Gap", w: 2.16 }
  ];
  let hx = 0.17;
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.17, y: 0.65, w: 9.66, h: 0.34,
    fill: { color: C.navy }, line: { color: C.navy }
  });
  hdr.forEach(h => {
    s.addText(h.text, {
      x: hx + 0.02, y: 0.65, w: h.w - 0.04, h: 0.34,
      fontFace: FONT, fontSize: 9, bold: true, color: C.white,
      valign: "middle", margin: 0
    });
    hx += h.w;
  });

  const tableData = [
    ["[4]", "2025", "AV-HuBERT + ViViT Face Encoder", "IEEE Trans. Human-Machine Syst.", "FakeAVCeleb", "Acc: 99.29%", "Extremely high resource demands (249M params, 140 ms latency). Unsuitable for edge deployment. Operates as a black box — no explainability."],
    ["[3]", "2025", "Cross-modal alignment + LoRA/Adapters on frozen CLIP ViT and Whisper", "arXiv preprint", "FakeAVCeleb, IDForge", "Acc: 99.0% AUC: 99.96%", "Massive backbone footprint (~330M params). High computational overhead. Lacks any intrinsic explainability tracking during detection."],
    ["[2]", "2025", "Visual + LLM (CLIP-large, BLIP-large, LLaMA-3.2 11B) for NL reasoning", "ACM Int. Conf. Multimedia (MM)", "DF40", "AUC: 91.3%", "Natural language interpretability but extreme cost (>11B params). Inference takes ~28 s/sample. Not deployable on constrained devices."],
  ];

  tableData.forEach((row, i) => {
    const y = 1.02 + i * 1.38;
    const bg = i % 2 === 0 ? C.accent : C.white;
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.17, y, w: 9.66, h: 1.32,
      fill: { color: bg }, line: { color: "D0D8E8", pt: 0.4 }
    });
    let cx = 0.17;
    const wids = [0.42, 0.36, 1.38, 1.28, 0.83, 0.7, 2.16];
    row.forEach((cell, j) => {
      const isBold = j === 0;
      const isGap = j === 6;
      s.addText(cell, {
        x: cx + 0.04, y: y + 0.04, w: wids[j] - 0.08, h: 1.24,
        fontFace: FONT, fontSize: j === 6 ? 8 : 8.5,
        bold: isBold, color: isGap ? C.warn : (j === 0 ? C.navy : C.near),
        valign: "middle", margin: 0, wrap: true
      });
      cx += wids[j];
    });
  });

  s.addText("Key Insight: These models achieve high accuracy but are computationally massive (249M–330M params), rendering real-time edge deployment impossible.", {
    x: 0.17, y: 5.15, w: 9.66, h: 0.25,
    fontFace: FONT, fontSize: 8.5, bold: true, color: C.teal,
    italic: true, margin: 0
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 8 ── LITERATURE REVIEW II  (2025 compression + 2022 earlier)
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  addHeader(s, "Literature Review — Compression & Earlier Work (2025–2022)");
  addFooter(s, "8 / 24");

  const hdr = [
    { text: "Ref", w: 0.42 }, { text: "Year", w: 0.36 }, { text: "Methods", w: 1.38 },
    { text: "Venue", w: 1.28 }, { text: "Dataset", w: 0.83 }, { text: "Perf.", w: 0.7 }, { text: "Research Gap", w: 2.16 }
  ];
  let hx = 0.17;
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.17, y: 0.65, w: 9.66, h: 0.34,
    fill: { color: C.teal }, line: { color: C.teal }
  });
  hdr.forEach(h => {
    s.addText(h.text, {
      x: hx + 0.02, y: 0.65, w: h.w - 0.04, h: 0.34,
      fontFace: FONT, fontSize: 9, bold: true, color: C.white,
      valign: "middle", margin: 0
    });
    hx += h.w;
  });

  const tableData = [
    ["[7]", "2025", "Knowledge Distillation + Structured Pruning + Quantization", "arXiv preprint", "Synthbuster", "Acc: >90%", "Visual-only framework. Significant accuracy drop on cross-domain tests. Entirely overlooks preservation of explainability during footprint reduction."],
    ["[9]", "2021", "KD + Convolutional Autoencoder (CAE) to distill GradCAM explanations", "IEEE Int. Conf. Big Data", "CIFAR-10", "Acc: 90.9%", "Requires an auxiliary generative model. Evaluated only on simple image classification. Not designed for multimodal cross-attention or deepfake domains."],
    ["[1]", "2022", "Multi-modal multi-scale transformers (RGB + Frequency domain fusion)", "ACM ICMR", "FaceForensics++ (HQ)", "Acc: 97.93%", "Unimodal approach — ignores audio entirely. High computational overhead due to EfficientNet-b4 + transformer blocks. Lacks any form of explainability."],
  ];

  tableData.forEach((row, i) => {
    const y = 1.02 + i * 1.38;
    const bg = i % 2 === 0 ? C.accent : C.white;
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.17, y, w: 9.66, h: 1.32,
      fill: { color: bg }, line: { color: "D0D8E8", pt: 0.4 }
    });
    let cx = 0.17;
    const wids = [0.42, 0.36, 1.38, 1.28, 0.83, 0.7, 2.16];
    row.forEach((cell, j) => {
      const isBold = j === 0;
      const isGap = j === 6;
      s.addText(cell, {
        x: cx + 0.04, y: y + 0.04, w: wids[j] - 0.08, h: 1.24,
        fontFace: FONT, fontSize: j === 6 ? 8 : 8.5,
        bold: isBold, color: isGap ? C.warn : (j === 0 ? C.navy : C.near),
        valign: "middle", margin: 0, wrap: true
      });
      cx += wids[j];
    });
  });

  s.addText("Key Insight: Compression work [7] ignores explainability; XAI distillation [9] is image-only. No prior work compresses a multimodal deepfake model while preserving cross-attention reasoning.", {
    x: 0.17, y: 5.15, w: 9.66, h: 0.25,
    fontFace: FONT, fontSize: 8.5, bold: true, color: C.teal,
    italic: true, margin: 0
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 9 ── LITERATURE REVIEW SYNTHESIS
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  addHeader(s, "Literature Review — Synthesis & Positioning");
  addFooter(s, "9 / 24");

  // 2x2 problem quadrant
  panelBox(s, 0.2, 0.65, 9.6, 0.38, C.navy, "Literature Gap Analysis — Three Unresolved Problems");

  const quads = [
    {
      title: "Problem A: Size vs. Accuracy",
      body: "Models with highest accuracy (AV-Lip-Sync+ 99.29%, CAD 99.0%) have 249M–330M parameters — making real-time edge deployment impossible.",
      color: C.warn
    },
    {
      title: "Problem B: Compression vs. Modality",
      body: "All existing compression baselines (Karathanasis et al., LightFakeDetect) operate on visual-only models. None handle the audio-video cross-modal fusion critical for deepfake timing artifacts.",
      color: C.navy
    },
    {
      title: "Problem C: Explainability vs. Efficiency",
      body: "Standard KD (Hinton et al., XDistillation) optimizes task accuracy, not reasoning fidelity. Compressed models silently learn spurious features, degrading forensic trustworthiness.",
      color: C.teal
    },
    {
      title: "PIN-Lite's Position",
      body: "First framework to simultaneously address all three problems: multimodal audio-visual compression with attention-aware KD, achieving 10.83× size reduction with maintained accuracy.",
      color: "1A6B3C"
    },
  ];

  quads.forEach((q, i) => {
    const x = 0.2 + (i % 2) * 4.85;
    const y = 1.1 + Math.floor(i / 2) * 2.05;
    accentCard(s, x, y, 4.7, 1.95, q.color, q.title, [q.body], 10.5, 9);
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 10 ── OBJECTIVES
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  addHeader(s, "Research Objectives — PIN-Lite");
  addFooter(s, "10 / 24");

  const objs = [
    {
      num: "O1",
      title: "Multi-Stage Compression Pipeline",
      text: "Design and implement a systematic Distill → Prune → Quantize pipeline that compresses a multimodal audio-visual deepfake detector from 57.32 MB to under 6 MB without accuracy loss."
    },
    {
      num: "O2",
      title: "Lightweight Student Architecture",
      text: "Develop a MobileNetV3-based student model with reduced embedding dimension, attention heads, and transformer depth that maintains architectural compatibility for cross-attention map comparison with the teacher."
    },
    {
      num: "O3",
      title: "Attention-Aware Knowledge Distillation",
      text: "Formulate a three-component distillation loss (L_hard + L_soft + L_attn) that explicitly transfers audio-visual reasoning patterns from teacher to student via attention-map alignment."
    },
    {
      num: "O4",
      title: "Accurate & Deployable Deepfake Detection",
      text: "Achieve comparable or improved detection accuracy (F1 ≥ 0.98, AUC ≥ 0.95) on the LAV-DF benchmark while reducing inference latency by at least 2× and enabling CPU-based INT8 deployment."
    },
    {
      num: "O5",
      title: "Generalizability",
      text: "Examine how video resolution, audio quality, and manipulation techniques impact detection reliability across multimodal fusion pipelines, and validate the framework's cross-domain behavior."
    },
  ];

  objs.forEach((o, i) => {
    const x = i < 3 ? 0.2 : 0.2 + (i - 3) * 4.85;
    const y = i < 3 ? 0.68 + i * 1.52 : 5.02;
    const w = i < 3 ? 9.6 : 4.7;
    const h = i < 3 ? 1.42 : 0.8;

    if (i < 5) {
      s.addShape(pres.shapes.RECTANGLE, {
        x, y, w, h, fill: { color: C.lgray }, line: { color: "D0D8E8", pt: 0.5 }
      });
      // num badge
      s.addShape(pres.shapes.RECTANGLE, {
        x, y, w: 0.5, h,
        fill: { color: i < 3 ? C.navy : C.teal }, line: { color: i < 3 ? C.navy : C.teal }
      });
      s.addText(o.num, {
        x, y, w: 0.5, h,
        fontFace: FONT, fontSize: 11, bold: true, color: C.white,
        align: "center", valign: "middle", margin: 0
      });
      s.addText(o.title, {
        x: x + 0.57, y: y + 0.08, w: w - 0.65, h: 0.28,
        fontFace: FONT, fontSize: 10, bold: true, color: C.navy, margin: 0
      });
      const ty = y + 0.38;
      s.addText(o.text, {
        x: x + 0.57, y: ty, w: w - 0.65, h: h - ty + y - 0.08,
        fontFace: FONT, fontSize: 9, color: C.near, margin: 0, wrap: true
      });
    }
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 11 ── DATASET
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  addHeader(s, "Datasets — Training & Evaluation");
  addFooter(s, "11 / 24");

  // LAV-DF
  panelBox(s, 0.2, 0.65, 9.6, 0.38, C.navy, "Primary Dataset: LAV-DF (Localized Audio-Visual Deepfake Dataset)");
  accentCard(s, 0.2, 1.1, 9.6, 1.55, C.navy,
    "LAV-DF — Large-Scale Audio-Visual Deepfake",
    [
      "100,000+ clips covering facial reenactment, face swapping, and audio synthesis across varied indoor/outdoor scenarios.",
      "Frame-level annotations enable both detection and temporal forgery localization tasks.",
      "Split: Training 3,400 samples | Test 1,550 samples (1,145 fake / 405 real).",
      "Focuses on lip-sync variations and visual quality changes — the primary benchmark for PIN-Lite."
    ], 10.5, 9);

  // FakeAVCeleb
  panelBox(s, 0.2, 2.75, 9.6, 0.38, C.teal, "Secondary Dataset: FakeAVCeleb");
  accentCard(s, 0.2, 3.2, 9.6, 0.95, C.teal,
    "FakeAVCeleb — Multi-Method Audio-Visual Fake Videos",
    [
      "19,500 fake videos generated via multiple methods: Faceswap, FSGAN, and Wav2Lip.",
      "Used for cross-dataset zero-shot evaluation to assess generalization behavior."
    ], 10.5, 9);

  // Sample attributes table
  panelBox(s, 0.2, 4.22, 9.6, 0.32, "163560", "Common Sample Preprocessing");
  const attrs = [
    ["Video Frames", "30 frames per sample, resized to 128 × 128"],
    ["Audio Features", "13 MFCC features computed with 25 ms windows, 10 ms hop; T_a = 60 frames"],
    ["Labels", "Binary (Real / Fake) — balanced for training"],
  ];
  attrs.forEach(([k, v], i) => {
    const bg = i % 2 === 0 ? C.accent : C.white;
    const y = 4.58 + i * 0.24;
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.2, y, w: 9.6, h: 0.24, fill: { color: bg }, line: { color: "D0D8E8", pt: 0.4 }
    });
    s.addText(k, {
      x: 0.28, y: y + 0.02, w: 2.0, h: 0.2,
      fontFace: FONT, fontSize: 8.5, bold: true, color: C.navy, margin: 0
    });
    s.addText(v, {
      x: 2.38, y: y + 0.02, w: 7.3, h: 0.2,
      fontFace: FONT, fontSize: 8.5, color: C.near, margin: 0
    });
  });

  // Figure placeholder
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.2, y: 5.1, w: 4.65, h: 0.0
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 12 ── PIN-LITE FRAMEWORK OVERVIEW
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  addHeader(s, "Proposed PIN-Lite Framework — Overview");
  addFooter(s, "12 / 24");

  // Three-stage pipeline overview
  const stages = [
    {
      num: "Phase 1", title: "Knowledge Distillation",
      sub: "Attention-Aware",
      body: "A lightweight MobileNetV3-based student learns from PinPoint (teacher) using a 3-component loss: hard labels (BCE), soft logits (KL-Div, T=2.0), and attention map transfer (MSE on cross-attention maps).",
      color: C.navy
    },
    {
      num: "Phase 2", title: "Iterative Pruning",
      sub: "ℓ₁-Norm Unstructured",
      body: "3 rounds × 20% target sparsity on Linear and GRU weights. Each round followed by 3 fine-tuning recovery epochs. Concentrates pruning in cross-attention heads.",
      color: C.teal
    },
    {
      num: "Phase 3", title: "Post-Training Quantization",
      sub: "INT8 / FP16",
      body: "Dynamic INT8 quantization on all nn.Linear and nn.GRU layers (no calibration data required). Alternative: FP16 mixed-precision via amp.autocast for GPU-equipped edge devices.",
      color: "1A6B3C"
    },
  ];

  stages.forEach((st, i) => {
    const x = 0.2 + i * 3.25;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 0.68, w: 3.1, h: 0.38,
      fill: { color: st.color }, line: { color: st.color }
    });
    s.addText(`${st.num} — ${st.title}`, {
      x: x + 0.05, y: 0.68, w: 3.0, h: 0.38,
      fontFace: FONT, fontSize: 10, bold: true, color: C.white,
      align: "center", valign: "middle", margin: 0
    });
    s.addText(st.sub, {
      x, y: 1.1, w: 3.1, h: 0.28,
      fontFace: FONT, fontSize: 9, bold: true, color: st.color,
      align: "center", margin: 0
    });
    accentCard(s, x, 1.4, 3.1, 1.55, st.color, null, [st.body], 9, 8.8);

    // Arrow between stages
    if (i < 2) {
      s.addShape(pres.shapes.RECTANGLE, {
        x: x + 3.1, y: 1.1, w: 0.14, h: 0.75,
        fill: { color: C.mgray }, line: { color: C.mgray }
      });
      s.addText("→", {
        x: x + 3.06, y: 1.15, w: 0.22, h: 0.65,
        fontFace: FONT, fontSize: 16, color: C.mgray, align: "center"
      });
    }
  });

  // Metrics summary row
  panelBox(s, 0.2, 3.08, 9.6, 0.35, C.navy, "Progressive Pipeline Results — Key Numbers");
  const metrics = [
    ["Teacher", "57.32 MB", "15.0M params", "98.62 ms"],
    ["+ Distilled", "6.62 MB", "1.69M params", "45.93 ms"],
    ["+ Pruned", "6.62 MB", "1.69M params", "44.18 ms"],
    ["+ Quantized (INT8)", "5.29 MB", "1.22M params", "171 ms†"],
  ];
  metrics.forEach((m, i) => {
    const x = 0.2 + i * 2.4;
    const bg = i === 3 ? C.teal : (i % 2 === 0 ? C.accent : C.white);
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 3.46, w: 2.4, h: 1.65,
      fill: { color: bg }, line: { color: "D0D8E8", pt: 0.5 }
    });
    s.addText(m[0], {
      x: x + 0.05, y: 3.5, w: 2.3, h: 0.32,
      fontFace: FONT, fontSize: 10, bold: true,
      color: i === 3 ? C.white : C.navy, align: "center", margin: 0
    });
    [m[1], m[2], m[3]].forEach((v, j) => {
      s.addText(v, {
        x: x + 0.05, y: 3.86 + j * 0.38, w: 2.3, h: 0.35,
        fontFace: FONT, fontSize: 9.5,
        color: i === 3 ? "C8F0E8" : C.near, align: "center", margin: 0
      });
    });
  });

  // Placeholder for framework figure
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.2, y: 5.15, w: 9.6, h: 0.28,
    fill: { color: C.lgray }, line: { color: "D0D8E8", pt: 0.4 }
  });
  s.addText("† CPU inference (INT8). All other latencies measured on GPU.  |  [FIGURE PLACEHOLDER: Fig. 1 — PIN-Lite Framework Overview Diagram (from paper)]", {
    x: 0.25, y: 5.17, w: 9.5, h: 0.24,
    fontFace: FONT, fontSize: 8, color: C.mgray, italic: true, margin: 0
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 13 ── ARCHITECTURAL EVOLUTION: TEACHER → STUDENT
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  addHeader(s, "Architectural Evolution — Teacher to Student");
  addFooter(s, "13 / 24");

  // Figure placeholder (main)
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.2, y: 0.65, w: 9.6, h: 3.35,
    fill: { color: "F0F4FA" }, line: { color: "B0C4DE", pt: 1.0 }
  });
  s.addText("[FIGURE PLACEHOLDER]\nTeacher–Student Architecture Comparison Diagram\n(Fig. from PIN-Lite paper — Architectural Evolution slide)\n\nLeft side: PinPoint Teacher (ResNet-18 backbone, 3× GCA blocks H=8, d=256)\nRight side: PIN-Lite Student (MobileNetV3-Small, 2× GCA blocks H=4, d=128)\nCenter: Attention Transfer (L_attn = MSE)", {
    x: 0.2, y: 0.65, w: 9.6, h: 3.35,
    fontFace: FONT, fontSize: 9.5, color: C.mgray,
    align: "center", valign: "middle"
  });

  // Comparison table below figure
  const cols = [
    { label: "Component", teacher: "Teacher (PinPoint)", student: "Student (PIN-Lite)" },
    { label: "Video Backbone", teacher: "ResNet-18 (frozen first 6 layers)", student: "MobileNetV3-Small (frozen first 3 blocks)" },
    { label: "Embedding Dim", teacher: "d = 256", student: "d_s = 128" },
    { label: "Attention Heads", teacher: "H = 8", student: "H_s = 4" },
    { label: "Cross-Attn Layers", teacher: "L = 3", student: "L_s = 2" },
    { label: "Parameters", teacher: "15.0M", student: "1.69M (−88.7%)" },
    { label: "Disk Size", teacher: "57.32 MB", student: "6.62 MB (8.66×)" },
  ];

  panelBox(s, 0.2, 4.08, 9.6, 0.35, "163560", "Architectural Comparison");

  cols.forEach((row, i) => {
    const bg = i === 0 ? C.navy : (i % 2 === 1 ? C.accent : C.white);
    const y = 4.46 + i * 0.24;
    s.addShape(pres.shapes.RECTANGLE, { x: 0.2, y, w: 9.6, h: 0.24, fill: { color: bg }, line: { color: "D0D8E8", pt: 0.4 } });
    [[0.2, 3.1, row.label], [3.35, 3.1, row.teacher], [6.5, 3.3, row.student]].forEach(([x, w, txt]) => {
      s.addText(txt, {
        x: x + 0.06, y: y + 0.02, w: w - 0.1, h: 0.21,
        fontFace: FONT, fontSize: 8.5, bold: i === 0,
        color: i === 0 ? C.white : (txt === row.student && i > 0 ? C.teal : C.near),
        valign: "middle", margin: 0
      });
    });
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 14 ── PHASE 1: ATTENTION-AWARE KNOWLEDGE DISTILLATION
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  addHeader(s, "Phase 1 — Attention-Aware Knowledge Distillation");
  addFooter(s, "14 / 24");

  panelBox(s, 0.2, 0.65, 9.6, 0.35, C.navy, "Three-Component Distillation Loss");

  // Loss equation
  s.addText("L_total = α · L_hard  +  (1−α) · L_soft  +  β · L_attn", {
    x: 0.2, y: 1.06, w: 9.6, h: 0.5,
    fontFace: "Courier New", fontSize: 15, bold: true, color: C.navy,
    align: "center", valign: "middle"
  });

  // Three loss boxes
  const losses = [
    {
      label: "L_hard",
      title: "Hard Label Loss (BCE)",
      body: "Binary cross-entropy between student's classification logits and ground-truth labels y. Provides direct task supervision ensuring the student learns the classification boundary.",
      params: "Default α = 0.5 | Optimal α = 0.3",
      color: C.navy
    },
    {
      label: "L_soft",
      title: "Soft Distillation Loss (KL-Div)",
      body: "KL divergence between temperature-softened logits of teacher and student. Transfers 'dark knowledge' encoded in the teacher's output distribution. T² scaling compensates for gradient reduction.",
      params: "Temperature T = 2.0 | Weight (1−α)",
      color: C.teal
    },
    {
      label: "L_attn",
      title: "Attention Transfer Loss (MSE)",
      body: "Mean squared error between teacher and student cross-attention maps. Explicitly forces the student to inherit audio-visual reasoning patterns. Bilinear interpolation aligns maps when spatial dims differ.",
      params: "Default β = 5.0 | Optimal β = 3.0",
      color: "1A6B3C"
    },
  ];

  losses.forEach((l, i) => {
    const x = 0.2 + i * 3.25;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 1.65, w: 3.1, h: 0.35,
      fill: { color: l.color }, line: { color: l.color }
    });
    s.addText(l.label, {
      x, y: 1.65, w: 3.1, h: 0.35,
      fontFace: "Courier New", fontSize: 12, bold: true, color: C.white,
      align: "center", valign: "middle", margin: 0
    });
    s.addText(l.title, {
      x, y: 2.04, w: 3.1, h: 0.28,
      fontFace: FONT, fontSize: 9.5, bold: true, color: l.color,
      align: "center", margin: 0
    });
    accentCard(s, x, 2.35, 3.1, 1.88, l.color, null, [l.body, "", l.params], 9, 8.8);
  });

  // Student training protocol
  panelBox(s, 0.2, 4.3, 9.6, 0.32, "163560", "Student Training Protocol");
  const proto = [
    ["Optimizer", "AdamW  |  lr = 2×10⁻⁴  |  weight decay = 10⁻⁴  |  Cosine Annealing scheduler"],
    ["Training Duration", "20 epochs full training; 10 epochs for ablation studies"],
    ["Batch Size / Augmentation", "Batch = 8  |  Gaussian noise + color jittering + temporal shuffling to prevent shortcut learning"],
  ];
  proto.forEach(([k, v], i) => {
    const bg = i % 2 === 0 ? C.accent : C.white;
    const y = 4.65 + i * 0.22;
    s.addShape(pres.shapes.RECTANGLE, { x: 0.2, y, w: 9.6, h: 0.22, fill: { color: bg }, line: { color: "D0D8E8", pt: 0.4 } });
    s.addText(k, { x: 0.28, y: y + 0.02, w: 2.1, h: 0.18, fontFace: FONT, fontSize: 8, bold: true, color: C.navy, margin: 0 });
    s.addText(v, { x: 2.45, y: y + 0.02, w: 7.2, h: 0.18, fontFace: FONT, fontSize: 8, color: C.near, margin: 0 });
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 15 ── PHASE 2: ITERATIVE PRUNING
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  addHeader(s, "Phase 2 — ℓ₁-Norm Iterative Pruning");
  addFooter(s, "15 / 24");

  // Left: process description
  panelBox(s, 0.2, 0.65, 5.6, 0.35, C.teal, "Pruning Process");

  const steps = [
    { n: "1", t: "Target Layers", b: "Global ℓ₁-norm unstructured pruning applied to all nn.Linear and nn.GRU weights. MobileNetV3 Conv2d layers are excluded due to high pre-trained weight magnitudes." },
    { n: "2", t: "Iterative Schedule", b: "R = 3 rounds × p = 20% target sparsity per round. Each round zeroes out the p-fraction of weights with the smallest ℓ₁ magnitudes." },
    { n: "3", t: "Fine-Tuning Recovery", b: "After each pruning round, the model is fine-tuned for E_ft = 3 epochs on the training set using only L_hard (BCE) to recover any accuracy drop." },
    { n: "4", t: "Permanent Masks", b: "After the final round, pruning masks are made permanent via prune.remove(), collapsing masks into weight tensors for deployment." },
  ];

  steps.forEach((st, i) => {
    const y = 1.07 + i * 0.98;
    s.addShape(pres.shapes.RECTANGLE, { x: 0.2, y, w: 0.4, h: 0.9, fill: { color: C.teal }, line: { color: C.teal } });
    s.addText(st.n, { x: 0.2, y, w: 0.4, h: 0.9, fontFace: FONT, fontSize: 14, bold: true, color: C.white, align: "center", valign: "middle", margin: 0 });
    s.addShape(pres.shapes.RECTANGLE, { x: 0.65, y, w: 5.1, h: 0.9, fill: { color: C.lgray }, line: { color: "D0D8E8", pt: 0.4 } });
    s.addText(st.t, { x: 0.72, y: y + 0.06, w: 4.97, h: 0.25, fontFace: FONT, fontSize: 9.5, bold: true, color: C.teal, margin: 0 });
    s.addText(st.b, { x: 0.72, y: y + 0.34, w: 4.97, h: 0.52, fontFace: FONT, fontSize: 8.5, color: C.near, margin: 0, wrap: true });
  });

  // Right: Key insight + sparsity breakdown
  panelBox(s, 6.0, 0.65, 3.8, 0.35, C.navy, "Key Insight on Pruning");

  accentCard(s, 6.0, 1.07, 3.8, 1.4, C.warn,
    "Why Effective Sparsity is Low",
    [
      "MobileNetV3 backbone weights resist ℓ₁ selection — pre-trained high magnitudes are never the smallest.",
      "Pruning concentrates entirely in cross-attention projection heads.",
      "Primary compression comes from KD (88.7% param reduction) and Quantization — not pruning."
    ], 9.5, 8.5);

  // Sparsity breakdown table
  panelBox(s, 6.0, 2.55, 3.8, 0.32, C.navy, "Per-Layer Sparsity Breakdown");
  const sparsity = [
    ["MobileNetV3 (Conv2d)", "54.7%", "0.0%"],
    ["Audio CNN (Conv1d)", "1.6%", "0.0%"],
    ["Audio GRU", "5.8%", "0.0%"],
    ["Cross-Attn (MHA)", "7.8%", "18.6%"],
    ["Self-Attn (MHA)", "7.8%", "0.0%"],
    ["Gate + FFN (Linear)", "17.6%", "0.0%"],
    ["Projection + Heads", "4.5%", "0.0%"],
  ];
  // header
  ["Layer Category", "% of Model", "Sparsity"].forEach((h, j) => {
    s.addText(h, {
      x: 6.0 + j * 1.27, y: 2.9, w: 1.27, h: 0.25,
      fontFace: FONT, fontSize: 7.5, bold: true, color: C.navy, margin: 0
    });
  });
  sparsity.forEach((row, i) => {
    const bg = i % 2 === 0 ? C.accent : C.white;
    const y = 3.18 + i * 0.265;
    s.addShape(pres.shapes.RECTANGLE, { x: 6.0, y, w: 3.8, h: 0.265, fill: { color: bg }, line: { color: "D0D8E8", pt: 0.3 } });
    row.forEach((cell, j) => {
      const isSparsity = j === 2;
      s.addText(cell, {
        x: 6.0 + j * 1.27 + 0.04, y: y + 0.03, w: 1.19, h: 0.22,
        fontFace: FONT, fontSize: 7.5,
        bold: isSparsity && cell !== "0.0%",
        color: isSparsity && cell !== "0.0%" ? C.teal : C.near,
        margin: 0
      });
    });
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 16 ── PHASE 3: POST-TRAINING QUANTIZATION
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  addHeader(s, "Phase 3 — Post-Training Quantization");
  addFooter(s, "16 / 24");

  panelBox(s, 0.2, 0.65, 4.7, 0.35, C.navy, "Dynamic INT8 Quantization");
  panelBox(s, 5.1, 0.65, 4.7, 0.35, C.teal, "FP16 Mixed-Precision (Alternative)");

  // INT8 details
  const int8 = [
    ["Applied To", "All nn.Linear and nn.GRU layers"],
    ["Mechanism", "FP32 weights → INT8; activations quantized dynamically at runtime"],
    ["Calibration", "No calibration data required — fully post-training"],
    ["Size Reduction", "6.62 MB → 5.29 MB (1.25× further reduction)"],
    ["Best Suited For", "CPU-only or NPU-equipped mobile platforms"],
  ];
  int8.forEach(([k, v], i) => {
    const bg = i % 2 === 0 ? C.accent : C.white;
    const y = 1.06 + i * 0.35;
    s.addShape(pres.shapes.RECTANGLE, { x: 0.2, y, w: 4.7, h: 0.35, fill: { color: bg }, line: { color: "D0D8E8", pt: 0.4 } });
    s.addText(k, { x: 0.28, y: y + 0.04, w: 1.7, h: 0.27, fontFace: FONT, fontSize: 8.5, bold: true, color: C.navy, margin: 0 });
    s.addText(v, { x: 2.0, y: y + 0.04, w: 2.82, h: 0.27, fontFace: FONT, fontSize: 8.5, color: C.near, margin: 0, wrap: true });
  });

  // FP16 details
  const fp16 = [
    ["Applied To", "All model weights cast to FP16"],
    ["Mechanism", "Inference via amp.autocast; LayerNorm retained in FP32 for stability"],
    ["Size Reduction", "6.62 MB → 3.31 MB (2× reduction; 17.32× vs teacher)"],
    ["Latency", "37.93 ms on GPU — fastest GPU speed of all variants"],
    ["Best Suited For", "GPU-equipped edge devices (e.g., Jetson Nano with FP16 CUDA cores)"],
  ];
  fp16.forEach(([k, v], i) => {
    const bg = i % 2 === 0 ? C.accent : C.white;
    const y = 1.06 + i * 0.35;
    s.addShape(pres.shapes.RECTANGLE, { x: 5.1, y, w: 4.7, h: 0.35, fill: { color: bg }, line: { color: "D0D8E8", pt: 0.4 } });
    s.addText(k, { x: 5.18, y: y + 0.04, w: 1.7, h: 0.27, fontFace: FONT, fontSize: 8.5, bold: true, color: C.teal, margin: 0 });
    s.addText(v, { x: 6.9, y: y + 0.04, w: 2.82, h: 0.27, fontFace: FONT, fontSize: 8.5, color: C.near, margin: 0, wrap: true });
  });

  // Pipeline summary
  panelBox(s, 0.2, 2.9, 9.6, 0.35, C.navy, "Pipeline Summary — Teacher to Final Quantized Student");

  const summary = [
    ["Teacher (PinPoint)", "57.32 MB", "15.0M", "98.62 ms (GPU)", "97.37%", "–"],
    ["+ Distilled (PIN-Lite)", "6.62 MB", "1.69M", "45.93 ms (GPU)", "97.53%", "8.66×"],
    ["+ Pruned", "6.62 MB", "1.69M", "44.18 ms (GPU)", "97.40%", "8.66×"],
    ["INT8 (Final)", "5.29 MB", "1.22M", "171 ms† (CPU)", "98.22%", "10.83×"],
    ["FP16 (Alt.)", "3.31 MB", "1.69M", "37.93 ms (GPU)", "97.50%", "17.32×"],
  ];
  const sHdr = ["Model Stage", "Size", "Params", "Latency", "Accuracy", "Size↓"];
  sHdr.forEach((h, j) => {
    const widths = [2.4, 1.1, 1.0, 1.65, 1.15, 1.15];
    const x = 0.2 + widths.slice(0, j).reduce((a, b) => a + b, 0);
    s.addText(h, {
      x: x + 0.04, y: 3.28, w: widths[j] - 0.08, h: 0.25,
      fontFace: FONT, fontSize: 8.5, bold: true, color: C.navy, margin: 0
    });
  });
  summary.forEach((row, i) => {
    const widths = [2.4, 1.1, 1.0, 1.65, 1.15, 1.15];
    const bg = i === 3 ? "E8F4EC" : (i === 4 ? "E8F0F8" : (i % 2 === 0 ? C.white : C.accent));
    const y = 3.57 + i * 0.32;
    let cx = 0.2;
    row.forEach((cell, j) => {
      s.addShape(pres.shapes.RECTANGLE, { x: cx, y, w: widths[j], h: 0.32, fill: { color: bg }, line: { color: "D0D8E8", pt: 0.4 } });
      s.addText(cell, {
        x: cx + 0.04, y: y + 0.04, w: widths[j] - 0.08, h: 0.24,
        fontFace: FONT, fontSize: 8.5,
        bold: (i === 3 || i === 4) && j === 0,
        color: i === 3 ? "1A6B3C" : (i === 4 ? C.teal : C.near),
        margin: 0
      });
      cx += widths[j];
    });
  });
  s.addText("† CPU inference (INT8). All other latencies measured on GPU (NVIDIA T4/P100).", {
    x: 0.2, y: 5.17, w: 9.6, h: 0.22, fontFace: FONT, fontSize: 7.5, color: C.mgray, italic: true, margin: 0
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 17 ── EXPERIMENTAL SETUP
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  addHeader(s, "Experimental Setup");
  addFooter(s, "17 / 24");

  panelBox(s, 0.2, 0.65, 4.7, 0.35, C.navy, "Software Stack");
  const sw = [
    ["Framework", "PyTorch + GradScaler (Mixed Precision Training)"],
    ["Quantization", "PyTorch Dynamic Quantization (INT8 / FP16 Post-Training)"],
    ["Language", "Python 3.x"],
    ["Notebooks", "Kaggle Notebooks (T4 & P100 GPU access)"],
  ];
  sw.forEach(([k, v], i) => {
    const bg = i % 2 === 0 ? C.accent : C.white;
    const y = 1.06 + i * 0.35;
    s.addShape(pres.shapes.RECTANGLE, { x: 0.2, y, w: 4.7, h: 0.35, fill: { color: bg }, line: { color: "D0D8E8", pt: 0.4 } });
    s.addText(k, { x: 0.28, y: y + 0.04, w: 1.55, h: 0.27, fontFace: FONT, fontSize: 9, bold: true, color: C.navy, margin: 0 });
    s.addText(v, { x: 1.85, y: y + 0.04, w: 2.97, h: 0.27, fontFace: FONT, fontSize: 9, color: C.near, margin: 0 });
  });

  panelBox(s, 0.2, 2.5, 4.7, 0.35, C.teal, "Hardware");
  const hw = [
    ["Primary GPU", "NVIDIA T4 (16 GB VRAM)"],
    ["Secondary GPU", "NVIDIA P100 (16 GB VRAM)"],
    ["Platform", "Kaggle Notebooks (cloud execution)"],
  ];
  hw.forEach(([k, v], i) => {
    const bg = i % 2 === 0 ? C.accent : C.white;
    const y = 2.91 + i * 0.35;
    s.addShape(pres.shapes.RECTANGLE, { x: 0.2, y, w: 4.7, h: 0.35, fill: { color: bg }, line: { color: "D0D8E8", pt: 0.4 } });
    s.addText(k, { x: 0.28, y: y + 0.04, w: 1.55, h: 0.27, fontFace: FONT, fontSize: 9, bold: true, color: C.teal, margin: 0 });
    s.addText(v, { x: 1.85, y: y + 0.04, w: 2.97, h: 0.27, fontFace: FONT, fontSize: 9, color: C.near, margin: 0 });
  });

  // Hyperparameter table
  panelBox(s, 5.1, 0.65, 4.7, 0.35, C.navy, "Training Hyperparameters");
  const hpA = [
    ["Optimizer", "AdamW"],
    ["Weight Decay", "10⁻⁴"],
    ["Learning Rate", "2 × 10⁻⁴"],
    ["Scheduler", "Cosine Annealing"],
    ["Epochs (Full Training)", "20"],
    ["Epochs (Ablation)", "10"],
    ["Batch Size", "8"],
  ];
  hpA.forEach(([k, v], i) => {
    const bg = i % 2 === 0 ? C.accent : C.white;
    const y = 1.06 + i * 0.3;
    s.addShape(pres.shapes.RECTANGLE, { x: 5.1, y, w: 4.7, h: 0.3, fill: { color: bg }, line: { color: "D0D8E8", pt: 0.4 } });
    s.addText(k, { x: 5.18, y: y + 0.03, w: 2.5, h: 0.24, fontFace: FONT, fontSize: 9, bold: true, color: C.navy, margin: 0 });
    s.addText(v, { x: 7.7, y: y + 0.03, w: 1.98, h: 0.24, fontFace: FONT, fontSize: 9, color: C.near, margin: 0 });
  });

  panelBox(s, 5.1, 3.16, 4.7, 0.35, C.teal, "Distillation Hyperparameters");
  const hpB = [
    ["Distillation Temperature (T)", "2.0"],
    ["Hard/Soft Balance (α)", "0.5"],
    ["Attention Transfer Weight (β)", "5.0"],
    ["Student CA Layers (L_s)", "2"],
  ];
  hpB.forEach(([k, v], i) => {
    const bg = i % 2 === 0 ? C.accent : C.white;
    const y = 3.57 + i * 0.35;
    s.addShape(pres.shapes.RECTANGLE, { x: 5.1, y, w: 4.7, h: 0.35, fill: { color: bg }, line: { color: "D0D8E8", pt: 0.4 } });
    s.addText(k, { x: 5.18, y: y + 0.04, w: 3.0, h: 0.27, fontFace: FONT, fontSize: 9, bold: true, color: C.teal, margin: 0 });
    s.addText(v, { x: 8.2, y: y + 0.04, w: 1.48, h: 0.27, fontFace: FONT, fontSize: 9, color: C.near, margin: 0 });
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 18 ── MAIN RESULTS
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  addHeader(s, "Main Results — LAV-DF Test Set");
  addFooter(s, "18 / 24");

  panelBox(s, 0.2, 0.65, 9.6, 0.35, C.navy, "Progressive Optimization Results on LAV-DF Test Set (N = 1,550)");

  const hdrCols = ["Model", "Size (MB)", "Params", "Lat. (ms)", "Accuracy", "Precision", "Recall", "F1", "AUC"];
  const widths = [2.3, 1.0, 0.85, 0.85, 0.95, 0.85, 0.75, 0.7, 0.65];
  let hx = 0.2;
  s.addShape(pres.shapes.RECTANGLE, { x: 0.2, y: 1.04, w: 9.6, h: 0.3, fill: { color: "163560" }, line: { color: "163560" } });
  hdrCols.forEach((h, j) => {
    s.addText(h, {
      x: hx + 0.03, y: 1.04, w: widths[j] - 0.06, h: 0.3,
      fontFace: FONT, fontSize: 8.5, bold: true, color: C.white, valign: "middle", margin: 0
    });
    hx += widths[j];
  });

  const tableRows = [
    ["Teacher (PinPoint)", "57.32", "15.0M", "98.62", "97.37%", "0.984", "0.980", "0.982", "0.968"],
    ["Distilled (Student)", "6.62", "1.69M", "45.93", "97.53%", "0.973", "0.994", "0.983", "0.958"],
    ["+ Pruned", "6.62", "1.69M", "44.18", "97.40%", "0.971", "0.994", "0.982", "0.956"],
    ["PIN-Lite (INT8)", "5.29", "1.22M", "171†", "98.22%", "0.984", "0.992", "0.988", "0.973"],
  ];

  tableRows.forEach((row, i) => {
    const isHighlight = i === 3;
    const bg = isHighlight ? "E8F4EC" : (i % 2 === 0 ? C.white : C.accent);
    const y = 1.37 + i * 0.52;
    let cx = 0.2;
    row.forEach((cell, j) => {
      s.addShape(pres.shapes.RECTANGLE, { x: cx, y, w: widths[j], h: 0.52, fill: { color: bg }, line: { color: "D0D8E8", pt: 0.4 } });
      s.addText(cell, {
        x: cx + 0.04, y: y + 0.08, w: widths[j] - 0.08, h: 0.36,
        fontFace: FONT, fontSize: j === 0 ? 9.5 : 10,
        bold: isHighlight,
        color: isHighlight ? "1A6B3C" : (j === 0 ? C.navy : C.near),
        valign: "middle", margin: 0
      });
      cx += widths[j];
    });
  });

  // Key takeaway callouts
  const callouts = [
    { val: "10.83×", lbl: "Size Reduction", sub: "57.32 → 5.29 MB", color: C.navy },
    { val: "12.30×", lbl: "Param Reduction", sub: "15.0M → 1.22M", color: C.teal },
    { val: "+0.85%", lbl: "Accuracy Gain", sub: "97.37% → 98.22%", color: "1A6B3C" },
    { val: "2.15×", lbl: "GPU Speedup", sub: "98.62 → 45.93 ms", color: C.warn },
  ];

  callouts.forEach((c, i) => {
    const x = 0.2 + i * 2.42;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 3.62, w: 2.32, h: 1.45,
      fill: { color: C.navy }, line: { color: C.navy }
    });
    s.addText(c.val, {
      x, y: 3.7, w: 2.32, h: 0.62,
      fontFace: FONT, fontSize: 30, bold: true, color: C.white, align: "center", margin: 0
    });
    s.addText(c.lbl, {
      x, y: 4.34, w: 2.32, h: 0.28,
      fontFace: FONT, fontSize: 9, bold: true, color: "A0C4E8", align: "center", margin: 0
    });
    s.addText(c.sub, {
      x, y: 4.65, w: 2.32, h: 0.24,
      fontFace: FONT, fontSize: 8, color: C.mgray, align: "center", margin: 0
    });
  });

  s.addText("† CPU inference (INT8). All other latencies measured on GPU.", {
    x: 0.2, y: 5.15, w: 9.6, h: 0.22, fontFace: FONT, fontSize: 7.5, color: C.mgray, italic: true, margin: 0
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 19 ── COMPARATIVE ANALYSIS
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  addHeader(s, "Comparative Analysis — PIN-Lite vs. State of the Art");
  addFooter(s, "19 / 24");

  panelBox(s, 0.2, 0.65, 9.6, 0.35, C.navy, "Comparison of Multimodal Deepfake Detection and Compression Methods");

  // Table
  const cHdr = ["Method", "Year", "Modality", "Params (M)", "Best Acc.", "Dataset", "Compression", "Explainability"];
  const cWidths = [1.9, 0.5, 1.0, 0.85, 0.82, 0.95, 1.5, 1.78];
  let chx = 0.2;
  s.addShape(pres.shapes.RECTANGLE, { x: 0.2, y: 1.04, w: 9.6, h: 0.3, fill: { color: "163560" }, line: { color: "163560" } });
  cHdr.forEach((h, j) => {
    s.addText(h, { x: chx + 0.03, y: 1.04, w: cWidths[j] - 0.06, h: 0.3, fontFace: FONT, fontSize: 8, bold: true, color: C.white, valign: "middle", margin: 0 });
    chx += cWidths[j];
  });

  const cRows = [
    ["XDistillation [10]", "2021", "Visual", "3.5", "90.9%", "CIFAR-10", "KD + CAE", "GradCAM (image only)"],
    ["M2TR [1]", "2022", "RGB+Freq.", "~19+", "97.93%", "FF++ (HQ)", "None", "None"],
    ["DF-P2E [2]", "2025", "Visual+LLM", ">11,000", "91.3% AUC", "DF40", "4-bit LLM", "Textual (NL)"],
    ["CAD [3]", "2025", "Audio-Visual", "~330", "99.0%", "FakeAVCeleb", "LoRA/Adapters", "None"],
    ["AV-Lip-Sync+ [4]", "2025", "Audio-Visual", "249", "99.29%", "FakeAVCeleb", "None", "None"],
    ["Karathanasis [7]", "2025", "Visual", "~4.5", ">90%", "Synthbuster", "KD+Prune+Quant", "None"],
    ["PinPoint (Ours-W1)", "2025", "Audio-Visual", "15.0", "97.47%", "LAV-DF+FAV", "None", "IG, SHAP, TCAV, CAV"],
    ["PIN-Lite (Ours-W2)", "2026", "Audio-Visual", "1.22–1.69", "98.22%", "LAV-DF", "KD+Prune+Quant", "Cross-Attn Map Fidelity"],
  ];

  cRows.forEach((row, i) => {
    const isOurs = i >= 6;
    const isPinLite = i === 7;
    const bg = isPinLite ? "E8F4EC" : (isOurs ? C.accent : (i % 2 === 0 ? C.white : C.accent));
    const y = 1.37 + i * 0.47;
    let cx = 0.2;
    row.forEach((cell, j) => {
      s.addShape(pres.shapes.RECTANGLE, { x: cx, y, w: cWidths[j], h: 0.47, fill: { color: bg }, line: { color: "D0D8E8", pt: 0.4 } });
      s.addText(cell, {
        x: cx + 0.04, y: y + 0.04, w: cWidths[j] - 0.08, h: 0.39,
        fontFace: FONT, fontSize: j === 0 ? 8 : 8,
        bold: isPinLite,
        color: isPinLite ? "1A6B3C" : (isOurs ? C.navy : C.near),
        valign: "middle", margin: 0, wrap: true
      });
      cx += cWidths[j];
    });
  });

  s.addText("PIN-Lite is the only method that jointly addresses multimodal audio-visual compression AND cross-attention reasoning fidelity at 1.22M parameters.", {
    x: 0.2, y: 5.15, w: 9.6, h: 0.22,
    fontFace: FONT, fontSize: 8.5, bold: true, color: C.teal, italic: true, margin: 0
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 20 ── CONCLUSION
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  addHeader(s, "Conclusion — Summary of Contributions");
  addFooter(s, "20 / 24");

  panelBox(s, 0.2, 0.65, 9.6, 0.35, C.navy, "PIN-Lite — Three Principal Contributions");

  const contribs = [
    {
      n: "C1", title: "First Multimodal Compression Framework",
      body: "Attention-aware KD + iterative pruning + post-training quantization applied to a multimodal audio-visual deepfake detector — the first pipeline to jointly optimize efficiency and cross-modal reasoning fidelity.",
      color: C.navy
    },
    {
      n: "C2", title: "Lightweight Student Architecture",
      body: "MobileNetV3-based student with reduced attention heads (H=4), embedding dimension (d=128), and transformer depth (L=2) — achieving 88.7% parameter reduction while maintaining architectural compatibility for attention comparison.",
      color: C.teal
    },
    {
      n: "C3", title: "Significant Footprint Reduction with Accuracy Gain",
      body: "10.83× model size reduction (57.32 MB → 5.29 MB), 12.30× parameter reduction (15.0M → 1.22M), and 2.15× GPU speedup — with detection accuracy improving from 97.37% to 98.22% on the LAV-DF benchmark.",
      color: "1A6B3C"
    },
  ];

  contribs.forEach((c, i) => {
    const y = 1.06 + i * 1.35;
    s.addShape(pres.shapes.RECTANGLE, { x: 0.2, y, w: 9.6, h: 1.25, fill: { color: C.lgray }, line: { color: "D0D8E8", pt: 0.5 } });
    s.addShape(pres.shapes.RECTANGLE, { x: 0.2, y, w: 0.55, h: 1.25, fill: { color: c.color }, line: { color: c.color } });
    s.addText(c.n, { x: 0.2, y, w: 0.55, h: 1.25, fontFace: FONT, fontSize: 13, bold: true, color: C.white, align: "center", valign: "middle", margin: 0 });
    s.addText(c.title, { x: 0.83, y: y + 0.1, w: 8.88, h: 0.32, fontFace: FONT, fontSize: 11, bold: true, color: c.color, margin: 0 });
    s.addText(c.body, { x: 0.83, y: y + 0.48, w: 8.88, h: 0.68, fontFace: FONT, fontSize: 9.5, color: C.near, margin: 0, wrap: true });
  });

  // Central message
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.2, y: 5.1, w: 9.6, h: 0.23, fill: { color: C.navy }, line: { color: C.navy }
  });
  s.addText("Central Message: A multimodal deepfake detector can be compressed 10.83× for edge deployment while improving accuracy from 97.37% to 98.22%.", {
    x: 0.25, y: 5.1, w: 9.5, h: 0.23,
    fontFace: FONT, fontSize: 9, bold: true, color: C.white, valign: "middle", margin: 0
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 21 ── FUTURE WORK
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  addHeader(s, "Future Work");
  addFooter(s, "21 / 24");

  panelBox(s, 0.2, 0.65, 4.65, 0.35, C.teal, "Short-Term Directions (Ongoing)");
  panelBox(s, 5.1, 0.65, 4.7, 0.35, C.navy, "Proposed Work 3 — Data Generation (Tentative)");

  const shortTerm = [
    ["Explainability Preservation Score (EPS)", "Formalize a novel architecture-agnostic metric combining Spearman's rank correlation and top-k IoU to quantify cross-attention map fidelity between teacher and student models."],
    ["Efficient Attention Variant Analysis", "Systematically evaluate Multi-Query, Low-Rank, and Linear Attention variants within the student architecture to identify optimal accuracy–explainability trade-offs."],
    ["Ablation Studies & Design Guidelines", "Conduct comprehensive ablations on distillation temperature, loss weights, network depth, and pruning rates to establish actionable design guidelines for multimodal transformer compression."],
    ["Structured Backbone Pruning", "EPS-guided channel/filter pruning of MobileNetV3 (currently 0% sparsity, 54.7% of total params) for deeper compression."],
    ["Hardware Deployment", "TensorRT + ONNX Runtime on physical edge devices (NVIDIA Jetson, Qualcomm mobile) with measured power and latency benchmarks."],
  ];
  shortTerm.forEach(([h, b], i) => {
    accentCard(s, 0.2, 1.06 + i * 0.88, 4.65, 0.82, C.teal, h, [b], 9, 8);
  });

  accentCard(s, 5.1, 1.06, 4.7, 2.0, C.navy,
    "Synthetic Audio-Visual Deepfake Dataset",
    [
      "Build a new benchmark with diverse cross-modal manipulation types to improve model generalization beyond academic datasets (LAV-DF, FakeAVCeleb).",
      "Controlled generation covering lip-sync, voice-swap, and identity-swap variations — including modern generative AI tools (Sora-class, latent diffusion).",
      "Aim: fill the gap between lab benchmarks and in-the-wild adversarial deepfakes."
    ], 10, 9);

  panelBox(s, 5.1, 3.13, 4.7, 0.35, C.navy, "Proposed Work 4 — Foundation Model (Tentative)");
  accentCard(s, 5.1, 3.55, 4.7, 1.55, C.navy,
    "Lightweight Foundation Model for Multimodal Deepfake Detection",
    [
      "Design a compact, inherently interpretable foundation model for audio-visual deepfake detection.",
      "Leverage insights from PIN-Lite compression to build a model that is both general-purpose and edge-deployable from the ground up, rather than compressing an existing large model."
    ], 10, 9);
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 22 ── REFERENCES I
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  addHeader(s, "References");
  addFooter(s, "22 / 24");

  const refs1 = [
    "[1]  J. Wang et al., \"M2TR: Multi-modal multi-scale transformers for deepfake detection,\" ACM Int. Conf. Multimedia Retrieval (ICMR), 2022.",
    "[2]  S. Tariq et al., \"From Prediction to Explanation: Multimodal, Explainable, and Interactive Deepfake Detection Framework for Non-Expert Users,\" ACM Int. Conf. Multimedia (MM), 2025.",
    "[3]  Y. Du et al., \"CAD: A general multimodal framework for video deepfake detection via cross-modal alignment and distillation,\" arXiv, 2025.",
    "[4]  S. A. Shahzad et al., \"AV-Lip-Sync+: Leveraging AV-HuBERT to exploit multimodal inconsistency for deepfake detection,\" IEEE Trans. Human-Machine Syst., 2025.",
    "[5]  M. Javed et al., \"Audio-Visual Synchronization and Lip Movement Analysis for Real-Time Deepfake Detection,\" Int. J. Comput. Intell. Syst., 2025.",
    "[6]  H. H. Raval, M. S. Patel, S. D. Parmar, \"A Review on Explainable AI for Deepfake Detection Leveraging Hybrid Deep Learning Techniques,\" SPU-JSTMR, 2025.",
    "[7]  A. Karathanasis et al., \"A brief review for compression and transfer learning techniques in deepfake detection,\" arXiv, 2025.",
    "[8]  Z. Cai et al., \"Do you really mean that? Content driven audio-visual deepfake dataset and multimodal method for temporal forgery localization,\" IEEE DSP, 2022.",
    "[9]  N. Shazeer, \"Fast transformer decoding: One write-head is all you need,\" arXiv, 2019.",
    "[10] R. Alharbi, M. N. Vu, M. T. Thai, \"Learning interpretation with explainable knowledge distillation,\" IEEE Int. Conf. Big Data, 2021.",
  ];

  refs1.forEach((ref, i) => {
    s.addText(ref, {
      x: 0.2, y: 0.68 + i * 0.458,
      w: 9.6, h: 0.42,
      fontFace: FONT, fontSize: 8.5, color: C.near,
      valign: "top", margin: 0, wrap: true
    });
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 23 ── REFERENCES II
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  addHeader(s, "References (continued)");
  addFooter(s, "23 / 24");

  const refs2 = [
    "[11] A. Katharopoulos et al., \"Transformers are RNNs: Fast autoregressive transformers with linear attention,\" ICML, 2020.",
    "[12] A. Howard et al., \"Searching for MobileNetV3,\" IEEE ICCV, 2019.",
    "[13] G. Hinton, O. Vinyals, J. Dean, \"Distilling the knowledge in a neural network,\" arXiv, 2015.",
    "[14] K. He et al., \"Deep residual learning for image recognition,\" IEEE CVPR, 2016.",
    "[15] M. Raza, K. M. Malik, \"MultimodalTrace: Deepfake Detection using Audiovisual Representation Learning,\" IEEE/CVF CVPR Workshops, 2023.",
    "[16] V. Hondru et al., \"ExDDV: A New Dataset for Explainable Deepfake Detection in Video,\" arXiv, 2025.",
    "[17] C. Yu et al., \"Explicit Correlation Learning for Generalizable Cross-Modal Deepfake Detection,\" IEEE ICME, 2024.",
    "[18] H. Ilyas, A. Javed, K. M. Malik, \"ConvNext-PNet: An interpretable and explainable deep-learning model for deepfakes detection,\" IEEE IJCB, 2024.",
    "[19] B. Jacob et al., \"Quantization and training of neural networks for efficient integer-arithmetic-only inference,\" IEEE CVPR, 2018.",
    "[20] N. Malik et al., \"Interpretability-aware pruning for efficient medical image analysis,\" arXiv, 2025.",
  ];

  refs2.forEach((ref, i) => {
    s.addText(ref, {
      x: 0.2, y: 0.68 + i * 0.458,
      w: 9.6, h: 0.42,
      fontFace: FONT, fontSize: 8.5, color: C.near,
      valign: "top", margin: 0, wrap: true
    });
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 24 ── THANK YOU
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.navy };

  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.08, fill: { color: C.teal }, line: { color: C.teal } });
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 5.545, w: 10, h: 0.08, fill: { color: C.teal }, line: { color: C.teal } });

  s.addText("Thank You", {
    x: 0.5, y: 0.9, w: 9, h: 1.3,
    fontFace: FONT, fontSize: 54, bold: true, color: C.white, align: "center"
  });

  // Divider
  s.addShape(pres.shapes.RECTANGLE, {
    x: 2.5, y: 2.35, w: 5, h: 0.05, fill: { color: C.teal }, line: { color: C.teal }
  });

  s.addText("Palak Parmar  |  24RCP002  |  CSE-SoT, PDEU", {
    x: 0.5, y: 2.5, w: 9, h: 0.38, fontFace: FONT, fontSize: 14, color: "A0C4E8", align: "center"
  });
  s.addText("24rcp002@sot.pdpu.ac.in", {
    x: 0.5, y: 2.9, w: 9, h: 0.3, fontFace: FONT, fontSize: 12, color: C.mgray, align: "center"
  });

  s.addText("PIN-Lite: Lightweight Multimodal Deepfake Detection with Explainability Preservation", {
    x: 0.5, y: 3.4, w: 9, h: 0.4,
    fontFace: FONT, fontSize: 11, color: "7BA7CC", align: "center", italic: true
  });

  const people2 = [
    { label: "Supervisor", name: "Dr. SantoshKumar Bharti", info: "CSE-SoT, PDEU" },
    { label: "External Supervisor", name: "Dr. Chintan Bhatt", info: "University of Wollongong" },
  ];
  people2.forEach((p, i) => {
    const x = 1.8 + i * 4.5;
    s.addShape(pres.shapes.RECTANGLE, { x, y: 4.0, w: 3.5, h: 1.1, fill: { color: "162B50" }, line: { color: "2B5EA7", pt: 0.7 } });
    s.addText(p.label, { x: x + 0.1, y: 4.06, w: 3.3, h: 0.25, fontFace: FONT, fontSize: 8.5, bold: true, color: "A0C4E8", align: "center", margin: 0 });
    s.addText(p.name, { x: x + 0.1, y: 4.33, w: 3.3, h: 0.3, fontFace: FONT, fontSize: 10, bold: true, color: C.white, align: "center", margin: 0 });
    s.addText(p.info, { x: x + 0.1, y: 4.65, w: 3.3, h: 0.3, fontFace: FONT, fontSize: 9, color: "C0D8F0", align: "center", margin: 0 });
  });
}

// ── WRITE ───────────────────────────────────────────────────────────────────
pres.writeFile({ fileName: "5th_DC_PINLite_Redesigned-2.pptx" })
  .then(() => console.log("✅ Presentation written."))
  .catch(err => console.error("❌ Error:", err));