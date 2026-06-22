"""
====================================================================================
GAQ — Cross-dataset generalization (LAV-DF -> FakeAVCeleb)
====================================================================================
Open question from the literature (Karathanasis et al.): does compression degrade
cross-domain generalization more than in-domain accuracy? We answer it directly for
gated cross-attention by comparing, on FakeAVCeleb (zero-shot, trained on LAV-DF):

  * FP32 distilled student
  * GAQ-INT8 (ours)

If GAQ's cross-dataset AUC tracks the FP32 student's, quantization is "free" for
generalization — a cheap, novel result.

Run (after gaq_ptq.py / gaq_qat.py so a calibrated GAQ checkpoint exists):
    python gaq_crossdataset.py
Outputs: gaq_crossdataset.csv
====================================================================================
"""

import json
import os

import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import gaq_config as C
import gaq_core as G


class FakeAVCelebDataset(Dataset):
    def __init__(self, metadata_path, data_dir, config, split="test"):
        self.config = config
        self.samples = []
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"FAVC metadata not found: {metadata_path}")
        with open(metadata_path) as f:
            meta = json.load(f)
        for item in [m for m in meta if m.get("split") == split]:
            vp = os.path.join(data_dir, item["preprocessed_video_path"])
            ap = os.path.join(data_dir, item["preprocessed_audio_path"])
            if os.path.exists(vp) and os.path.exists(ap):
                self.samples.append({"video_path": vp, "audio_path": ap,
                                     "label": 1.0 if item["label"] == "fake" else 0.0})
        print(f"FakeAVCeleb: {len(self.samples)} samples")
        self.normalize = T.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            s = self.samples[idx]
            video = torch.load(s["video_path"]).float() / 255.0
            audio = torch.load(s["audio_path"])
            if audio.ndim == 3:
                if audio.shape[0] == self.config.NUM_MFCC:
                    audio = audio.mean(dim=1).transpose(0, 1)
                elif audio.shape[1] == self.config.NUM_MFCC:
                    audio = audio.mean(dim=0).transpose(0, 1)
                elif audio.shape[2] == self.config.NUM_MFCC:
                    audio = audio.mean(dim=1)
                else:
                    return None
            elif audio.ndim == 2:
                if audio.shape[0] == self.config.NUM_MFCC:
                    audio = audio.transpose(0, 1)
                elif audio.shape[1] != self.config.NUM_MFCC:
                    return None
            else:
                return None
            video = self.normalize(video)
            nf = self.config.NUM_FRAMES
            t = video.shape[0]
            if t > nf:
                idxs = torch.linspace(0, t - 1, nf).long()
                video = video[idxs]
            elif t < nf:
                video = torch.cat([video, torch.zeros(nf - t, *video.shape[1:])], dim=0)
            return {"video": video, "audio": audio,
                    "label": torch.tensor(s["label"], dtype=torch.float)}
        except Exception:
            return None


def favc_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    max_a = max(b["audio"].shape[0] for b in batch)
    videos, audios, labels = [], [], []
    for b in batch:
        a = b["audio"]
        if a.shape[0] < max_a:
            a = F.pad(a, (0, 0, 0, max_a - a.shape[0]))
        videos.append(b["video"])
        audios.append(a)
        labels.append(b["label"])
    return {"video": torch.stack(videos), "audio": torch.stack(audios),
            "label": torch.stack(labels)}


def _load_or_build_gaq(student, s_cfg, device):
    """Load calibrated GAQ (QAT preferred, then PTQ); else build+calibrate fresh."""
    gaq = G.build_quantizable_student(student, num_bits=8, per_channel=True,
                                      observer="percentile")
    for ck in ("gaq_int8_qat.pth", "gaq_int8_ptq.pth"):
        p = C.out(ck)
        if os.path.exists(p):
            print(f"  Loaded calibrated GAQ from {ck}")
            gaq.load_state_dict(torch.load(p, map_location="cpu"), strict=False)
            G.set_model_policy(gaq, "gaq")
            for fq in G._all_act_fq(gaq):
                if torch.isfinite(fq.scale).all():
                    fq.enabled, fq.calibrating = True, False
            G.set_model_policy(gaq, "gaq")
            gaq.to(device)
            return gaq
    print("  No calibrated checkpoint found — calibrating on LAV-DF train ...")
    loaders = G.make_loaders(s_cfg, splits=("train",), batch_size=C.CALIBRATION_BATCH_SIZE)
    cb = max(1, C.CALIBRATION_SAMPLES // C.CALIBRATION_BATCH_SIZE)
    G.calibrate(gaq, loaders["train"], device, cb, policy="gaq")
    gaq.to(device)
    return gaq


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    probs, labels = [], []
    for batch in loader:
        if batch is None:
            continue
        logits, _, _ = model(batch["video"].to(device), batch["audio"].to(device))
        p = torch.sigmoid(logits.float()).squeeze(1).cpu().numpy()
        probs.extend(p.tolist())
        labels.extend(batch["label"].numpy().tolist())
    if not labels:
        return {}
    preds = [1.0 if x > 0.5 else 0.0 for x in probs]
    auc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else 0.0
    return {"acc": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, zero_division=0), "auc": float(auc)}


def main():
    C.banner("GAQ — CROSS-DATASET (LAV-DF -> FakeAVCeleb)")
    G.set_seed(C.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n[1/3] Loading student + GAQ ...")
    student, s_cfg = G.load_student(device=device)
    gaq = _load_or_build_gaq(student, s_cfg, device)

    print("\n[2/3] FakeAVCeleb test set ...")
    ds = FakeAVCelebDataset(C.FAVC_METADATA_PATH, C.FAVC_DATA_DIRECTORY, s_cfg)
    if len(ds) == 0:
        raise SystemExit("No FakeAVCeleb samples found — check FAVC paths in gaq_config.py")
    loader = DataLoader(ds, batch_size=C.EVAL_BATCH_SIZE, shuffle=False, collate_fn=favc_collate)

    print("\n[3/3] Evaluating FP32 student vs GAQ-INT8 ...")
    fp32 = evaluate(student, loader, device)
    gaqm = evaluate(gaq, loader, device)
    print(f"  FP32 student : {fp32}")
    print(f"  GAQ-INT8     : {gaqm}")
    delta_auc = (gaqm.get("auc", 0) - fp32.get("auc", 0))
    print(f"  Δ AUC (GAQ - FP32) = {delta_auc:+.4f}")

    import csv
    with open(C.out("gaq_crossdataset.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "favc_acc", "favc_f1", "favc_auc"])
        w.writerow(["FP32 student", round(fp32.get("acc", 0), 4),
                    round(fp32.get("f1", 0), 4), round(fp32.get("auc", 0), 4)])
        w.writerow(["GAQ-INT8 (ours)", round(gaqm.get("acc", 0), 4),
                    round(gaqm.get("f1", 0), 4), round(gaqm.get("auc", 0), 4)])
    print(f"\nSaved -> {C.out('gaq_crossdataset.csv')}")


if __name__ == "__main__":
    main()
