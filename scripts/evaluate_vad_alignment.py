"""Part 2: VAD ground-truth alignment (IoU / F1) for Grad-CAM.

IMPORTANT CAVEAT: The "reference" labels here come from running the
open-source Silero VAD model (github.com/snakers4/silero-vad) on each
clean, pre-mix single-speaker source. This is a reference from another
VAD model, NOT hand-labeled ground truth. Silero VAD can itself be wrong
(missed onsets, merged pauses, etc.), so treat these IoU/F1 numbers as
"agreement with a second automatic VAD system", not an absolute ceiling
on correctness.

Pipeline:
1. Run Silero VAD on each clean pre-mix source -> frame-level speech mask
   at the same time resolution as the model's own VAD/CAM output (T=188
   frames for a 3.0s/16kHz mixture, hop=256 samples @ 512-pt STFT).
2. Compute the CAM (single-layer TCN.TCN.9.conv1d, finalized in Part 1)
   for the VAD-logit target, per speaker, on the *mixture*.
3. Threshold the (min-max normalized) CAM at {0.3, 0.5, 0.7} and at the
   F1-maximizing threshold, compute frame-level IoU/F1 against the Silero
   reference mask for that speaker.
4. Do the same thresholding for the network's own predicted VAD
   probability (already an output) as a ceiling/context comparison.
5. Aggregate mean +/- std across all pairs already fetched (35 pooled).
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

gradcam_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(gradcam_root))

import network.model as module_arch
from data.librispeech import SpeakerSampler, load_utterance
from gradcam import GradCAM

SAMPLE_RATE = 16000
TARGET_SECONDS = 3.0
N_FFT = 512
HOP_LENGTH = 256
SINGLE_LAYER = "TCN.TCN.9.conv1d"  # finalized in Part 1 (ensemble did not clearly help)
THRESHOLDS = [0.3, 0.5, 0.7]


def normalize_audio(audio):
    audio = audio.astype(np.float32)
    peak = np.max(np.abs(audio))
    return audio / max(peak, 1e-8) * 0.9


def load_and_pad(path, target_length):
    signal = load_utterance(path)[:target_length]
    padded = np.zeros(target_length, dtype=np.float32)
    padded[:len(signal)] = signal
    return padded


def num_frames_for_length(num_samples, n_fft=N_FFT, hop=HOP_LENGTH):
    # matches torch.stft(..., center=True) framing used elsewhere in this repo
    return 1 + num_samples // hop


def silero_reference_mask(vad_model, clean_signal, num_frames):
    """Frame-level speech-active reference mask from Silero VAD on the clean source."""
    wav_t = torch.from_numpy(clean_signal.astype(np.float32))
    timestamps = get_speech_timestamps(wav_t, vad_model, sampling_rate=SAMPLE_RATE, return_seconds=False)
    mask = np.zeros(num_frames, dtype=bool)
    for seg in timestamps:
        start_frame = seg["start"] // HOP_LENGTH
        end_frame = min(num_frames, seg["end"] // HOP_LENGTH + 1)
        mask[start_frame:end_frame] = True
    return mask


def minmax(values):
    values = np.asarray(values, dtype=np.float64)
    span = values.max() - values.min()
    if span > 1e-12:
        return (values - values.min()) / span
    return np.zeros_like(values)


def compute_iou_f1(pred_mask, ref_mask):
    pred_mask = pred_mask.astype(bool)
    ref_mask = ref_mask.astype(bool)
    tp = np.sum(pred_mask & ref_mask)
    fp = np.sum(pred_mask & ~ref_mask)
    fn = np.sum(~pred_mask & ref_mask)
    union = np.sum(pred_mask | ref_mask)
    iou = float(tp / union) if union > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return iou, f1


def best_f1_threshold(scores, ref_mask, candidates=None):
    if candidates is None:
        candidates = np.linspace(0.05, 0.95, 19)
    best_thr, best_f1_val = 0.5, -1.0
    for thr in candidates:
        _, f1 = compute_iou_f1(scores >= thr, ref_mask)
        if f1 > best_f1_val:
            best_f1_val = f1
            best_thr = float(thr)
    return best_thr, best_f1_val


def compute_cam(model, audio, speaker, index, device):
    input_audio = audio.clone().detach().to(device).float().requires_grad_(True)
    gradcam = GradCAM(model, SINGLE_LAYER, device)
    try:
        with torch.enable_grad():
            output = model(input_audio)
            _ = output
            target = model.vad_logits[0, speaker, index]
            target.backward()
            activations = gradcam.hook.activations
            gradients = gradcam.hook.gradients
            weights = gradients.mean(dim=tuple(range(2, gradients.ndim)), keepdim=True)
            raw = torch.relu((weights * activations).sum(dim=1)).detach().cpu().numpy()[0]
            return raw
    finally:
        gradcam.hook.remove_hooks()


def load_all_pairs():
    pairs = []
    sel_sampler = SpeakerSampler("data/librispeech_samples", seed=123)
    sel_ids = sel_sampler.speaker_ids
    for i in range(20):
        spk1, spk2 = sel_ids[2 * i], sel_ids[2 * i + 1]
        pairs.append((sel_sampler.sample_utterance(spk1), sel_sampler.sample_utterance(spk2), f"sel{i+1}"))

    hold_sampler = SpeakerSampler("data/librispeech_holdout", seed=123)
    hold_ids = hold_sampler.speaker_ids
    for i in range(len(hold_ids) // 2):
        spk1, spk2 = hold_ids[2 * i], hold_ids[2 * i + 1]
        pairs.append((hold_sampler.sample_utterance(spk1), hold_sampler.sample_utterance(spk2), f"hold{i+1}"))
    return pairs


def main():
    device = "cpu"
    config = json.loads((gradcam_root / "configs" / "config_with_vad.json").read_text())
    model = module_arch.SeparationModel(**config["arch"]["args"]).to(device).eval()
    checkpoint = torch.load(gradcam_root / "weights" / "model_with_vad.pth", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=True)

    print("[*] Loading Silero VAD reference model...")
    vad_model = load_silero_vad()

    pairs = load_all_pairs()
    target_length = int(TARGET_SECONDS * SAMPLE_RATE)
    num_frames = num_frames_for_length(target_length)
    print(f"[*] Evaluating VAD ground-truth alignment over {len(pairs)} pairs, {num_frames} frames/mixture")

    rows = []
    for idx, (p1, p2, tag) in enumerate(pairs):
        clean_signals = [load_and_pad(p, target_length) for p in (p1, p2)]
        mixture = normalize_audio(clean_signals[0] + clean_signals[1])
        audio = torch.from_numpy(mixture).unsqueeze(0)

        with torch.no_grad():
            output = model(audio.to(device))
            vad_probs = torch.sigmoid(model.vad_logits).detach().cpu().numpy()[0]
            vad_logits = model.vad_logits.detach().cpu().numpy()[0]

        for speaker in range(2):
            ref_mask = silero_reference_mask(vad_model, clean_signals[speaker], num_frames)
            if ref_mask.sum() == 0 or ref_mask.sum() == len(ref_mask):
                # Degenerate reference (all-silence or all-speech): IoU/F1 undefined/meaningless, skip this speaker.
                print(f"  Pair {idx+1:02d} ({tag}) speaker {speaker}: SKIPPED (degenerate Silero reference mask)")
                continue

            vad_idx = int(np.argmax(np.abs(vad_logits[speaker])))
            raw_cam = compute_cam(model, audio, speaker, vad_idx, device)
            cam_norm = minmax(raw_cam)
            cam_norm = cam_norm[:num_frames] if len(cam_norm) >= num_frames else np.pad(cam_norm, (0, num_frames - len(cam_norm)))

            netvad_scores = vad_probs[speaker][:num_frames] if len(vad_probs[speaker]) >= num_frames else np.pad(vad_probs[speaker], (0, num_frames - len(vad_probs[speaker])))

            row = {"pair_index": idx + 1, "tag": tag, "speaker": speaker, "utt": (p1.name if speaker == 0 else p2.name)}
            for thr in THRESHOLDS:
                iou, f1 = compute_iou_f1(cam_norm >= thr, ref_mask)
                row[f"cam_iou_thr{thr}"] = iou
                row[f"cam_f1_thr{thr}"] = f1
                iou_net, f1_net = compute_iou_f1(netvad_scores >= thr, ref_mask)
                row[f"netvad_iou_thr{thr}"] = iou_net
                row[f"netvad_f1_thr{thr}"] = f1_net

            best_thr_cam, best_f1_cam = best_f1_threshold(cam_norm, ref_mask)
            best_iou_cam, _ = compute_iou_f1(cam_norm >= best_thr_cam, ref_mask)
            row["cam_best_thr"] = best_thr_cam
            row["cam_best_f1"] = best_f1_cam
            row["cam_best_iou"] = best_iou_cam

            best_thr_net, best_f1_net = best_f1_threshold(netvad_scores, ref_mask)
            best_iou_net, _ = compute_iou_f1(netvad_scores >= best_thr_net, ref_mask)
            row["netvad_best_thr"] = best_thr_net
            row["netvad_best_f1"] = best_f1_net
            row["netvad_best_iou"] = best_iou_net

            rows.append(row)
            print(f"  Pair {idx+1:02d} ({tag}) speaker {speaker}: CAM best-F1={row['cam_best_f1']:.3f} (thr={row['cam_best_thr']:.2f})  "
                  f"NetVAD best-F1={row['netvad_best_f1']:.3f} (thr={row['netvad_best_thr']:.2f})")

    output_dir = gradcam_root / "results" / "vad_alignment"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "vad_alignment_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "num_speaker_instances": len(rows),
        "target_layer": SINGLE_LAYER,
        "reference_source": "Silero VAD (github.com/snakers4/silero-vad) run on clean pre-mix sources -- "
                             "NOT hand-labeled ground truth, a second automatic VAD model's reference",
        "default_thresholds": {},
        "best_f1_threshold": {},
    }
    for thr in THRESHOLDS:
        summary["default_thresholds"][str(thr)] = {
            "cam_iou_mean": float(np.mean([r[f"cam_iou_thr{thr}"] for r in rows])),
            "cam_iou_std": float(np.std([r[f"cam_iou_thr{thr}"] for r in rows])),
            "cam_f1_mean": float(np.mean([r[f"cam_f1_thr{thr}"] for r in rows])),
            "cam_f1_std": float(np.std([r[f"cam_f1_thr{thr}"] for r in rows])),
            "netvad_iou_mean": float(np.mean([r[f"netvad_iou_thr{thr}"] for r in rows])),
            "netvad_iou_std": float(np.std([r[f"netvad_iou_thr{thr}"] for r in rows])),
            "netvad_f1_mean": float(np.mean([r[f"netvad_f1_thr{thr}"] for r in rows])),
            "netvad_f1_std": float(np.std([r[f"netvad_f1_thr{thr}"] for r in rows])),
        }
    summary["best_f1_threshold"] = {
        "cam_f1_mean": float(np.mean([r["cam_best_f1"] for r in rows])),
        "cam_f1_std": float(np.std([r["cam_best_f1"] for r in rows])),
        "cam_iou_mean": float(np.mean([r["cam_best_iou"] for r in rows])),
        "cam_iou_std": float(np.std([r["cam_best_iou"] for r in rows])),
        "cam_avg_best_threshold": float(np.mean([r["cam_best_thr"] for r in rows])),
        "netvad_f1_mean": float(np.mean([r["netvad_best_f1"] for r in rows])),
        "netvad_f1_std": float(np.std([r["netvad_best_f1"] for r in rows])),
        "netvad_iou_mean": float(np.mean([r["netvad_best_iou"] for r in rows])),
        "netvad_iou_std": float(np.std([r["netvad_best_iou"] for r in rows])),
        "netvad_avg_best_threshold": float(np.mean([r["netvad_best_thr"] for r in rows])),
    }
    ratio_f1 = summary["best_f1_threshold"]["cam_f1_mean"] / summary["best_f1_threshold"]["netvad_f1_mean"] if summary["best_f1_threshold"]["netvad_f1_mean"] > 0 else float("nan")
    summary["cam_as_fraction_of_netvad_ceiling_f1_bestthr"] = float(ratio_f1)

    summary_path = output_dir / "vad_alignment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 70)
    print("VAD GROUND-TRUTH (SILERO REFERENCE) ALIGNMENT SUMMARY:")
    print("=" * 70)
    for thr in THRESHOLDS:
        d = summary["default_thresholds"][str(thr)]
        print(f"Threshold={thr}: CAM F1={d['cam_f1_mean']:.3f}±{d['cam_f1_std']:.3f}, IoU={d['cam_iou_mean']:.3f}±{d['cam_iou_std']:.3f} | "
              f"NetVAD F1={d['netvad_f1_mean']:.3f}±{d['netvad_f1_std']:.3f}, IoU={d['netvad_iou_mean']:.3f}±{d['netvad_iou_std']:.3f}")
    b = summary["best_f1_threshold"]
    print(f"Best-F1 threshold: CAM F1={b['cam_f1_mean']:.3f}±{b['cam_f1_std']:.3f} (avg thr={b['cam_avg_best_threshold']:.2f}) | "
          f"NetVAD F1={b['netvad_f1_mean']:.3f}±{b['netvad_f1_std']:.3f} (avg thr={b['netvad_avg_best_threshold']:.2f})")
    print(f"CAM achieves {ratio_f1*100:.1f}% of the network's own VAD-prediction F1 ceiling (best-F1 threshold).")
    print("=" * 70)
    print(f"[+] Saved: {csv_path}\n            {summary_path}")


if __name__ == "__main__":
    main()
