"""Part 1: Multi-layer ensemble Grad-CAM vs single-layer TCN.TCN.9.conv1d.

Layer choice (Step 1): reused the existing entropy-corrected layer_scores.json
ranking (no new arbitrary picks). Split 24 blocks into thirds and took the
top scorer in each third:
  - Early (blocks 0-7):   TCN.TCN.0.conv1d  (score +0.5247)
  - Middle (blocks 8-15): TCN.TCN.9.conv1d  (score +0.8931) -- same as the
    existing single-layer winner, confirmed from the data rather than assumed.
  - Late (blocks 16-23):  TCN.TCN.19.conv1d (score +0.2306)

Temporal alignment (Step 2): verified (not assumed) that all three layers
produce identical activation length (T=188) for the standard 3.0s/16kHz
mixture used throughout this repo, since the TCN's dilated convs use
padding that preserves sequence length. No interpolation was required for
this pipeline's fixed audio duration; the interpolation fallback below is
kept only as a safety net in case that ever changes.

Ensemble (Step 3): compute each layer's CAM with the existing method,
min-max normalize each individually (matching the repo's existing
convention), then average elementwise across the 3 layers.

Comparison (Step 4): run both ensemble and single-layer through the existing
MAE-vs-random-noise-control + Wilcoxon validation, on the existing pooled
35 pairs (20 selection-set + 15 held-out), and report which wins.
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import torch

gradcam_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(gradcam_root))

import network.model as module_arch
from data.librispeech import SpeakerSampler, load_utterance
from gradcam import GradCAM

SAMPLE_RATE = 16000
TARGET_SECONDS = 3.0

EARLY_LAYER = "TCN.TCN.0.conv1d"
MID_LAYER = "TCN.TCN.9.conv1d"
LATE_LAYER = "TCN.TCN.19.conv1d"
ENSEMBLE_LAYERS = [EARLY_LAYER, MID_LAYER, LATE_LAYER]
SINGLE_LAYER = MID_LAYER


def normalize_audio(audio):
    audio = audio.astype(np.float32)
    peak = np.max(np.abs(audio))
    return audio / max(peak, 1e-8) * 0.9


def prepare_mixture(paths):
    target_length = int(TARGET_SECONDS * SAMPLE_RATE)
    signals = []
    for path in paths:
        signal = load_utterance(path)[:target_length]
        padded = np.zeros(target_length, dtype=np.float32)
        padded[:len(signal)] = signal
        signals.append(padded)
    mixture = normalize_audio(signals[0] + signals[1])
    return torch.from_numpy(mixture).unsqueeze(0)


def minmax(values):
    values = np.asarray(values, dtype=np.float64)
    result = np.zeros_like(values)
    span = values.max() - values.min()
    if span > 1e-12:
        result = (values - values.min()) / span
    return result


def compute_single_layer_cam(model, audio, layer, target_kind, speaker, index, device):
    input_audio = audio.clone().detach().to(device).float().requires_grad_(True)
    gradcam = GradCAM(model, layer, device)
    try:
        with torch.enable_grad():
            output = model(input_audio)
            separated = output[0]
            target = (model.vad_logits[0, speaker, index] if target_kind == "vad_logit"
                      else separated[0, speaker, index].abs())
            target.backward()
            activations = gradcam.hook.activations
            gradients = gradcam.hook.gradients
            if activations is None or gradients is None:
                raise RuntimeError("Hook did not capture activations/gradients")
            weights = gradients.mean(dim=tuple(range(2, gradients.ndim)), keepdim=True)
            raw = torch.relu((weights * activations).sum(dim=1)).detach().cpu().numpy()[0]
            if not np.isfinite(raw).all() or raw.max() <= 0 or np.ptp(raw) <= 1e-12:
                raise RuntimeError("CAM is zero, non-finite, or constant")
            return raw
    finally:
        gradcam.hook.remove_hooks()


def compute_ensemble_cam(model, audio, target_kind, speaker, index, device):
    """Average of individually min-max normalized single-layer CAMs.

    Interpolates to a common length only if lengths differ (Step 2 safety net;
    not exercised in practice since all three layers were verified to match).
    """
    per_layer_cams = []
    lengths = []
    for layer in ENSEMBLE_LAYERS:
        raw = compute_single_layer_cam(model, audio, layer, target_kind, speaker, index, device)
        per_layer_cams.append(raw)
        lengths.append(len(raw))

    common_length = max(lengths)
    aligned = []
    for raw, length in zip(per_layer_cams, lengths):
        if length != common_length:
            x_old = np.linspace(0, 1, length)
            x_new = np.linspace(0, 1, common_length)
            raw = np.interp(x_new, x_old, raw)
        aligned.append(minmax(raw))

    ensemble = np.mean(np.stack(aligned, axis=0), axis=0)
    return ensemble, lengths


def compute_pair_metrics(left, right, pair_idx, seed_offset):
    ind_left = minmax(left)
    ind_right = minmax(right)
    rng = np.random.default_rng(seed=seed_offset + pair_idx)
    random_map = rng.random(left.shape)
    return {
        "ind_mae": float(np.mean(np.abs(ind_left - ind_right))),
        "random_mae": float(np.mean(np.abs(ind_left - random_map))),
    }


def compute_wilcoxon_stats(real_maes, random_maes):
    res = stats.wilcoxon(real_maes, random_maes)
    diffs = np.array(random_maes) - np.array(real_maes)
    abs_diffs = np.abs(diffs)
    ranks = stats.rankdata(abs_diffs)
    w_pos = np.sum(ranks[diffs > 0])
    w_neg = np.sum(ranks[diffs < 0])
    tot = w_pos + w_neg
    r_rank_biserial = float((w_pos - w_neg) / tot) if tot > 0 else 0.0
    return {"w_statistic": float(res.statistic), "p_value": float(res.pvalue), "rank_biserial_r": r_rank_biserial}


def load_all_pairs():
    """Load the existing pooled pairs: 20 selection-set + 15 held-out (both already fetched)."""
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

    pairs = load_all_pairs()
    print(f"[*] Comparing ensemble ({ENSEMBLE_LAYERS}) vs single-layer ({SINGLE_LAYER}) over {len(pairs)} pairs")

    rows = []
    for idx, (p1, p2, tag) in enumerate(pairs):
        audio = prepare_mixture([p1, p2])
        with torch.no_grad():
            output = model(audio.to(device))
            vad_logits = model.vad_logits.detach().cpu().numpy()[0]
            separated = output[0].detach().cpu().numpy()[0]

        row = {"pair_index": idx + 1, "tag": tag, "utt_0": p1.name, "utt_1": p2.name}
        for method_name, cam_fn in [("single", None), ("ensemble", None)]:
            cams_vad, cams_wave = [], []
            for speaker in range(2):
                vad_idx = int(np.argmax(np.abs(vad_logits[speaker])))
                wave_idx = int(np.argmax(np.abs(separated[speaker])))
                if method_name == "single":
                    vad_cam = compute_single_layer_cam(model, audio, SINGLE_LAYER, "vad_logit", speaker, vad_idx, device)
                    wave_cam = compute_single_layer_cam(model, audio, SINGLE_LAYER, "waveform", speaker, wave_idx, device)
                else:
                    vad_cam, lengths_vad = compute_ensemble_cam(model, audio, "vad_logit", speaker, vad_idx, device)
                    wave_cam, lengths_wave = compute_ensemble_cam(model, audio, "waveform", speaker, wave_idx, device)
                    if speaker == 0 and idx == 0:
                        row["_ensemble_layer_lengths_vad"] = lengths_vad
                        row["_ensemble_layer_lengths_wave"] = lengths_wave
                cams_vad.append(vad_cam)
                cams_wave.append(wave_cam)

            vad_m = compute_pair_metrics(cams_vad[0], cams_vad[1], idx, seed_offset=3000)
            wave_m = compute_pair_metrics(cams_wave[0], cams_wave[1], idx, seed_offset=3000)
            row[f"{method_name}_vad_mae"] = vad_m["ind_mae"]
            row[f"{method_name}_vad_random_mae"] = vad_m["random_mae"]
            row[f"{method_name}_wave_mae"] = wave_m["ind_mae"]
            row[f"{method_name}_wave_random_mae"] = wave_m["random_mae"]

        rows.append(row)
        print(f"  Pair {idx+1:02d} ({tag}): single VAD MAE={row['single_vad_mae']:.4f} (rand {row['single_vad_random_mae']:.4f}) | "
              f"ensemble VAD MAE={row['ensemble_vad_mae']:.4f} (rand {row['ensemble_vad_random_mae']:.4f})")

    # Aggregate stats for both methods, both targets
    summary = {"num_pairs": len(rows), "ensemble_layers": ENSEMBLE_LAYERS, "single_layer": SINGLE_LAYER}
    for method_name in ["single", "ensemble"]:
        summary[method_name] = {}
        for target in ["vad", "wave"]:
            real = [r[f"{method_name}_{target}_mae"] for r in rows]
            rand = [r[f"{method_name}_{target}_random_mae"] for r in rows]
            wstats = compute_wilcoxon_stats(real, rand)
            summary[method_name][target] = {
                "real_mae_mean": float(np.mean(real)), "real_mae_std": float(np.std(real)),
                "random_mae_mean": float(np.mean(rand)), "random_mae_std": float(np.std(rand)),
                "wilcoxon": wstats,
            }

    output_dir = gradcam_root / "results" / "ensemble_cam"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "ensemble_vs_single_results.csv"
    fieldnames = [k for k in rows[0].keys() if not k.startswith("_")]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({k: v for k, v in r.items() if not k.startswith("_")} for r in rows)

    summary_path = output_dir / "ensemble_vs_single_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    # Comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    for ax_idx, target in enumerate(["vad", "wave"]):
        ax = axes[ax_idx]
        single_mae = [r[f"single_{target}_mae"] for r in rows]
        ensemble_mae = [r[f"ensemble_{target}_mae"] for r in rows]
        for i in range(len(rows)):
            ax.plot([1, 2], [single_mae[i], ensemble_mae[i]], color="gray", alpha=0.4, marker="o", markersize=3)
        m1, s1 = np.mean(single_mae), np.std(single_mae)
        m2, s2 = np.mean(ensemble_mae), np.std(ensemble_mae)
        ax.errorbar([1], [m1], yerr=[s1], fmt="o", color="navy", capsize=6, markersize=8, label=f"Single ({m1:.3f}±{s1:.3f})")
        ax.errorbar([2], [m2], yerr=[s2], fmt="s", color="darkorange", capsize=6, markersize=8, label=f"Ensemble ({m2:.3f}±{s2:.3f})")
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Single-layer\n(TCN.TCN.9)", "Ensemble\n(Blocks 0,9,19)"])
        ax.set_ylabel("Speaker-vs-Speaker MAE (lower = more informative)")
        ax.set_title(f"{'VAD-logit' if target == 'vad' else 'Waveform'} CAM (N={len(rows)})")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.3)
    fig.suptitle("Ensemble vs Single-Layer Grad-CAM: Speaker-vs-Speaker MAE", fontsize=13, fontweight="bold")
    plot_path = output_dir / "ensemble_vs_single_plot.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\n" + "=" * 70)
    print("ENSEMBLE vs SINGLE-LAYER COMPARISON:")
    print("=" * 70)
    for target_name, target in [("VAD-Logit", "vad"), ("Waveform", "wave")]:
        s = summary["single"][target]
        e = summary["ensemble"][target]
        print(f"{target_name} CAM:")
        print(f"  Single ({SINGLE_LAYER}): MAE={s['real_mae_mean']:.4f}±{s['real_mae_std']:.4f} vs Rand={s['random_mae_mean']:.4f}±{s['random_mae_std']:.4f}, p={s['wilcoxon']['p_value']:.2e}, r={s['wilcoxon']['rank_biserial_r']:.3f}")
        print(f"  Ensemble (0,9,19):        MAE={e['real_mae_mean']:.4f}±{e['real_mae_std']:.4f} vs Rand={e['random_mae_mean']:.4f}±{e['random_mae_std']:.4f}, p={e['wilcoxon']['p_value']:.2e}, r={e['wilcoxon']['rank_biserial_r']:.3f}")
        gap_single = s['random_mae_mean'] - s['real_mae_mean']
        gap_ensemble = e['random_mae_mean'] - e['real_mae_mean']
        print(f"  Separation gap (random - real): single={gap_single:.4f}, ensemble={gap_ensemble:.4f} -> {'ENSEMBLE WIDER' if gap_ensemble > gap_single else 'SINGLE WIDER'}")
    print("=" * 70)
    print(f"[+] Saved: {csv_path}\n            {summary_path}\n            {plot_path}")


if __name__ == "__main__":
    main()
