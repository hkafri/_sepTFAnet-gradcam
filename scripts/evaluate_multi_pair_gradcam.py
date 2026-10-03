"""Evaluate Grad-CAM across 20 non-overlapping speaker pairs.

Computes:
1. VAD-logit and waveform CAMs using the selected layer (TCN.TCN.6.conv1d).
2. Independent min-max MAE, shared-scale MAE, and real-vs-random control MAE.
3. Mean ± std for each metric across all 20 pairs.
4. Paired Wilcoxon signed-rank test comparing real speaker-vs-speaker MAE vs real-vs-random-noise MAE.
5. Saves CSV results and a paired comparison plot.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import soundfile as sf
import torch

gradcam_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(gradcam_root))

import network.model as module_arch
from data.librispeech import SpeakerSampler, load_utterance
from gradcam import GradCAM


SAMPLE_RATE = 16000
TARGET_SECONDS = 3.0
NUM_PAIRS = 20


def load_selected_layer():
    scores_path = gradcam_root / "results" / "layer_selection" / "layer_scores.json"
    if scores_path.exists():
        data = json.loads(scores_path.read_text())
        return data.get("winning_layer", "TCN.TCN.6.conv1d")
    return "TCN.TCN.6.conv1d"


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
    return torch.from_numpy(mixture).unsqueeze(0), signals


def compute_cam(model, audio, target_layer, target_kind, speaker, index, device):
    input_audio = audio.clone().detach().to(device).float().requires_grad_(True)
    gradcam = GradCAM(model, target_layer, device)
    try:
        with torch.enable_grad():
            output = model(input_audio)
            separated = output[0]
            if target_kind == "vad_logit":
                target = model.vad_logits[0, speaker, index]
            else:
                target = separated[0, speaker, index].abs()
            target.backward()
            activations = gradcam.hook.activations
            gradients = gradcam.hook.gradients
            if activations is None or gradients is None:
                raise RuntimeError("Grad-CAM hook did not capture activations and gradients")
            weights = gradients.mean(dim=tuple(range(2, gradients.ndim)), keepdim=True)
            raw = torch.relu((weights * activations).sum(dim=1)).detach().cpu().numpy()[0]
            if not np.isfinite(raw).all() or raw.max() <= 0 or np.ptp(raw) <= 1e-12:
                raise RuntimeError("CAM is zero, non-finite, or constant")
            return raw, float(target.detach().cpu())
    finally:
        gradcam.hook.remove_hooks()


def minmax(values, scale=1.0):
    values = np.asarray(values, dtype=np.float64)
    result = np.zeros_like(values)
    span = values.max() - values.min()
    if span > 1e-12:
        result = (values - values.min()) / span
    return result * scale


def compute_pair_metrics(left, right, pair_idx, seed_offset=1000):
    ind_left = minmax(left)
    ind_right = minmax(right)
    shared_scale = max(float(left.max()), float(right.max()), 1e-12)
    shared_left = np.maximum(left, 0) / shared_scale
    shared_right = np.maximum(right, 0) / shared_scale

    # Reproducible random control per pair with non-colliding seed offset
    rng = np.random.default_rng(seed=seed_offset + pair_idx)
    random_map = rng.random(left.shape)

    return {
        "ind_mae": float(np.mean(np.abs(ind_left - ind_right))),
        "ind_max": float(np.max(np.abs(ind_left - ind_right))),
        "shared_mae": float(np.mean(np.abs(shared_left - shared_right))),
        "shared_max": float(np.max(np.abs(shared_left - shared_right))),
        "random_mae": float(np.mean(np.abs(ind_left - random_map))),
        "random_max": float(np.max(np.abs(ind_left - random_map))),
    }


def compute_wilcoxon_stats(real_maes, random_maes):
    diffs = np.array(random_maes) - np.array(real_maes)
    res = stats.wilcoxon(real_maes, random_maes)
    stat = res.statistic
    pvalue = res.pvalue

    # Rank-biserial correlation r = (W+ - W-) / (W+ + W-)
    abs_diffs = np.abs(diffs)
    ranks = stats.rankdata(abs_diffs)
    w_pos = np.sum(ranks[diffs > 0])
    w_neg = np.sum(ranks[diffs < 0])
    tot = w_pos + w_neg
    r_rank_biserial = float((w_pos - w_neg) / tot) if tot > 0 else 0.0

    n = len(diffs)
    z_approx = (stat - n * (n + 1) / 4) / np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    r_z = float(abs(z_approx) / np.sqrt(n))

    return {
        "w_statistic": float(stat),
        "p_value": float(pvalue),
        "rank_biserial_r": r_rank_biserial,
        "effect_size_r": r_z,
    }


def create_paired_comparison_plot(pair_results, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    kinds = [("vad", "VAD-Logit CAM"), ("wave", "Waveform-target CAM")]

    for ax_idx, (kind_key, title) in enumerate(kinds):
        ax = axes[ax_idx]
        real_maes = [p[f"{kind_key}_ind_mae"] for p in pair_results]
        rand_maes = [p[f"{kind_key}_random_mae"] for p in pair_results]
        n_pairs = len(real_maes)

        x_real, x_rand = 1, 2

        # Draw paired lines for each pair
        for i in range(n_pairs):
            ax.plot([x_real, x_rand], [real_maes[i], rand_maes[i]],
                    color="gray", alpha=0.5, linewidth=1.2, linestyle="-", marker="o", markersize=4)

        # Plot means and std error bars
        mean_real, std_real = np.mean(real_maes), np.std(real_maes)
        mean_rand, std_rand = np.mean(rand_maes), np.std(rand_maes)

        ax.errorbar([x_real], [mean_real], yerr=[std_real], fmt="o", color="navy",
                    linewidth=2.5, elinewidth=2.5, capsize=6, markersize=8, label=f"Real Spk-vs-Spk\n({mean_real:.3f} ± {std_real:.3f})")
        ax.errorbar([x_rand], [mean_rand], yerr=[std_rand], fmt="s", color="crimson",
                    linewidth=2.5, elinewidth=2.5, capsize=6, markersize=8, label=f"Real-vs-Random Control\n({mean_rand:.3f} ± {std_rand:.3f})")

        ax.set_xticks([x_real, x_rand])
        ax.set_xticklabels(["Real Speaker-vs-Speaker", "Real-vs-Random Control"], fontsize=11, fontweight="bold")
        ax.set_ylabel("Mean Absolute Error (MAE)", fontsize=11)
        ax.set_title(f"{title} (N = {n_pairs} Pairs)", fontsize=12, fontweight="bold")
        ax.set_ylim(0.0, 0.7)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="upper left", frameon=True)

    fig.suptitle("Paired Comparison: Speaker Attention Sensitivity vs Random Control", fontsize=13, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--librispeech-root", type=Path, default=Path("data/librispeech_samples"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-pairs", type=int, default=NUM_PAIRS)
    parser.add_argument("--seed-offset", type=int, default=1000)
    parser.add_argument("--target-layer", default=None)
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--plot-output", type=Path, default=None)
    args = parser.parse_args()

    target_layer = args.target_layer if args.target_layer else load_selected_layer()
    print(f"[*] Multi-Pair Evaluation using Target Layer: {target_layer} (Seed Offset: {args.seed_offset})")

    sampler = SpeakerSampler(str(args.librispeech_root), seed=123)
    speaker_ids = sampler.speaker_ids
    if len(speaker_ids) < args.num_pairs * 2:
        raise ValueError(f"Need at least {args.num_pairs * 2} speakers in {args.librispeech_root}")

    config_path = gradcam_root / "configs" / "config_with_vad.json"
    checkpoint_path = gradcam_root / "weights" / "model_with_vad.pth"
    config = json.loads(config_path.read_text())
    model = module_arch.SeparationModel(**config["arch"]["args"]).to(args.device).eval()
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=True)

    pair_results = []
    print(f"[*] Processing {args.num_pairs} non-overlapping speaker pairs from {args.librispeech_root}...")

    for i in range(args.num_pairs):
        spk1, spk2 = speaker_ids[2 * i], speaker_ids[2 * i + 1]
        p1 = sampler.sample_utterance(spk1)
        p2 = sampler.sample_utterance(spk2)
        paths = [p1, p2]

        audio, _ = prepare_mixture(paths)
        with torch.no_grad():
            output = model(audio.to(args.device))
            vad_logits = model.vad_logits.detach().cpu().numpy()[0]
            separated = output[0].detach().cpu().numpy()[0]

        cams = {"vad_logit": [], "waveform": []}
        for speaker in range(2):
            vad_idx = int(np.argmax(np.abs(vad_logits[speaker])))
            wave_idx = int(np.argmax(np.abs(separated[speaker])))

            vad_cam, _ = compute_cam(model, audio, target_layer, "vad_logit", speaker, vad_idx, args.device)
            wave_cam, _ = compute_cam(model, audio, target_layer, "waveform", speaker, wave_idx, args.device)

            cams["vad_logit"].append(vad_cam)
            cams["waveform"].append(wave_cam)

        vad_m = compute_pair_metrics(cams["vad_logit"][0], cams["vad_logit"][1], i, seed_offset=args.seed_offset)
        wave_m = compute_pair_metrics(cams["waveform"][0], cams["waveform"][1], i, seed_offset=args.seed_offset)

        p_res = {
            "pair_index": i + 1,
            "speaker_0": spk1,
            "speaker_1": spk2,
            "utt_0": p1.name,
            "utt_1": p2.name,
            "vad_ind_mae": vad_m["ind_mae"],
            "vad_ind_max": vad_m["ind_max"],
            "vad_shared_mae": vad_m["shared_mae"],
            "vad_shared_max": vad_m["shared_max"],
            "vad_random_mae": vad_m["random_mae"],
            "vad_random_max": vad_m["random_max"],
            "wave_ind_mae": wave_m["ind_mae"],
            "wave_ind_max": wave_m["ind_max"],
            "wave_shared_mae": wave_m["shared_mae"],
            "wave_shared_max": wave_m["shared_max"],
            "wave_random_mae": wave_m["random_mae"],
            "wave_random_max": wave_m["random_max"],
        }
        pair_results.append(p_res)
        print(f"  Pair {i+1:02d} ({spk1} vs {spk2}): VAD MAE={vad_m['ind_mae']:.4f} (vs Rand {vad_m['random_mae']:.4f}), Wave MAE={wave_m['ind_mae']:.4f} (vs Rand {wave_m['random_mae']:.4f})")

    # Save CSV
    csv_path = args.csv_output if args.csv_output else (gradcam_root / "results" / "librispeech_gradcam" / "multi_pair_results.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(pair_results[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pair_results)

    # Compute aggregate stats & Wilcoxon test
    vad_real_maes = [p["vad_ind_mae"] for p in pair_results]
    vad_rand_maes = [p["vad_random_mae"] for p in pair_results]
    vad_stats = compute_wilcoxon_stats(vad_real_maes, vad_rand_maes)

    wave_real_maes = [p["wave_ind_mae"] for p in pair_results]
    wave_rand_maes = [p["wave_random_mae"] for p in pair_results]
    wave_stats = compute_wilcoxon_stats(wave_real_maes, wave_rand_maes)

    summary = {
        "num_pairs": args.num_pairs,
        "target_layer": target_layer,
        "data_root": str(args.librispeech_root),
        "seed_offset": args.seed_offset,
        "vad_logit": {
            "real_mae_mean": float(np.mean(vad_real_maes)),
            "real_mae_std": float(np.std(vad_real_maes)),
            "shared_mae_mean": float(np.mean([p["vad_shared_mae"] for p in pair_results])),
            "shared_mae_std": float(np.std([p["vad_shared_mae"] for p in pair_results])),
            "random_control_mae_mean": float(np.mean(vad_rand_maes)),
            "random_control_mae_std": float(np.std(vad_rand_maes)),
            "wilcoxon": vad_stats,
        },
        "waveform": {
            "real_mae_mean": float(np.mean(wave_real_maes)),
            "real_mae_std": float(np.std(wave_real_maes)),
            "shared_mae_mean": float(np.mean([p["wave_shared_mae"] for p in pair_results])),
            "shared_mae_std": float(np.std([p["wave_shared_mae"] for p in pair_results])),
            "random_control_mae_mean": float(np.mean(wave_rand_maes)),
            "random_control_mae_std": float(np.std(wave_rand_maes)),
            "wilcoxon": wave_stats,
        },
    }

    summary_path = args.summary_output if args.summary_output else (gradcam_root / "results" / "librispeech_gradcam" / "multi_pair_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))

    plot_path = args.plot_output if args.plot_output else (gradcam_root / "results" / "librispeech_gradcam" / "paired_comparison_plot.png")
    create_paired_comparison_plot(pair_results, plot_path)

    print("\n" + "=" * 70)
    print("MULTI-PAIR AGGREGATE RESULTS SUMMARY:")
    print("=" * 70)
    print(f"VAD-Logit CAM (N={args.num_pairs} pairs):")
    print(f"  Real Speaker-vs-Speaker MAE: {summary['vad_logit']['real_mae_mean']:.4f} ± {summary['vad_logit']['real_mae_std']:.4f}")
    print(f"  Real-vs-Random Control MAE:  {summary['vad_logit']['random_control_mae_mean']:.4f} ± {summary['vad_logit']['random_control_mae_std']:.4f}")
    print(f"  Wilcoxon Statistic W:        {vad_stats['w_statistic']}, p-value: {vad_stats['p_value']:.2e}, Rank-Biserial r: {vad_stats['rank_biserial_r']:.3f}")
    print("-" * 70)
    print(f"Waveform-target CAM (N={args.num_pairs} pairs):")
    print(f"  Real Speaker-vs-Speaker MAE: {summary['waveform']['real_mae_mean']:.4f} ± {summary['waveform']['real_mae_std']:.4f}")
    print(f"  Real-vs-Random Control MAE:  {summary['waveform']['random_control_mae_mean']:.4f} ± {summary['waveform']['random_control_mae_std']:.4f}")
    print(f"  Wilcoxon Statistic W:        {wave_stats['w_statistic']}, p-value: {wave_stats['p_value']:.2e}, Rank-Biserial r: {wave_stats['rank_biserial_r']:.3f}")
    print("=" * 70)
    print(f"[+] Saved artifacts to:\n    - {csv_path}\n    - {summary_path}\n    - {plot_path}\n")


if __name__ == "__main__":
    main()
