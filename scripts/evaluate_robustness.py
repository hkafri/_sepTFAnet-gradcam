"""Part 3: Robustness to noise and reverberation.

Reuses the existing ~35 pooled pairs (20 selection-set + 15 held-out) --
this evaluates condition robustness, not sample-size scaling (that's Part 4).

Conditions:
  - clean                     (existing baseline, reported for comparison)
  - reverb_t60_0.2 / 0.4 / 0.6  (pyroomacoustics ShoeBox ISM, independent RIR
                                  per speaker -- two people in different room
                                  positions would not share an impulse response)
  - noise_snr_10 / 5 / 0      (babble-noise proxy built from 8 LEFTOVER,
                                  UNUSED LibriSpeech utterances -- i.e. utterances
                                  from the same fetched pool that were never
                                  selected as the "anchor" utterance for any of
                                  the 35 pairs. This is a low-effort proxy, NOT
                                  a real ambient-noise corpus; a corpus such as
                                  WHAM! would be a stronger version of this test.)
  - reverb_t60_0.4 + noise_snr_5   (representative combined condition)

For each condition: re-run the existing MAE-vs-random-control + Wilcoxon
comparison (single-layer TCN.TCN.9.conv1d, finalized in Part 1) AND the
Part-2 IoU/F1-vs-Silero-reference metric. The Silero reference mask is
always computed from the ORIGINAL CLEAN pre-mix source (not the degraded
one), since "when did this speaker actually talk" doesn't change because
of added reverb/noise -- only what the network sees changes.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pyroomacoustics as pra
import scipy.stats as stats
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

gradcam_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(gradcam_root))

import network.model as module_arch
from data.librispeech import SpeakerSampler, load_utterance
from gradcam import GradCAM

SAMPLE_RATE = 16000
TARGET_SECONDS = 3.0
TARGET_LENGTH = int(TARGET_SECONDS * SAMPLE_RATE)
N_FFT = 512
HOP_LENGTH = 256
SINGLE_LAYER = "TCN.TCN.9.conv1d"
T60_VALUES = [0.2, 0.4, 0.6]
SNR_VALUES = [10, 5, 0]


def normalize_audio(audio, target_peak=0.9):
    audio = audio.astype(np.float32)
    peak = np.max(np.abs(audio))
    return audio / max(peak, 1e-8) * target_peak


def load_and_pad(path, target_length=TARGET_LENGTH):
    signal = load_utterance(path)[:target_length]
    padded = np.zeros(target_length, dtype=np.float32)
    padded[:len(signal)] = signal
    return padded


def num_frames_for_length(num_samples, n_fft=N_FFT, hop=HOP_LENGTH):
    return 1 + num_samples // hop


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


def leftover_unused_paths(all_used_names):
    unused = []
    for root in ["data/librispeech_samples", "data/librispeech_holdout"]:
        for f in Path(root).glob("*.flac"):
            if f.name not in all_used_names:
                unused.append(f)
    return unused


def build_babble_track(unused_paths, rng, n_sources=8, length=TARGET_LENGTH):
    """Sum n_sources leftover utterances (a low-effort babble-noise proxy)."""
    chosen = rng.choice(len(unused_paths), size=min(n_sources, len(unused_paths)), replace=False)
    babble = np.zeros(length, dtype=np.float32)
    for idx in chosen:
        sig = load_and_pad(unused_paths[idx], length)
        babble += sig
    return normalize_audio(babble, target_peak=1.0)


def add_noise_at_snr(signal, noise, snr_db):
    sig_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    target_noise_power = sig_power / (10 ** (snr_db / 10.0))
    scale = np.sqrt(target_noise_power / max(noise_power, 1e-12))
    return signal + noise * scale


def simulate_reverb(signal, t60, src_pos, mic_pos, room_dim=(6.0, 5.0, 3.0), fs=SAMPLE_RATE):
    e_absorption, max_order = pra.inverse_sabine(t60, list(room_dim))
    room = pra.ShoeBox(list(room_dim), fs=fs, materials=pra.Material(e_absorption), max_order=max_order)
    room.add_source(list(src_pos), signal=signal.astype(np.float64))
    room.add_microphone(list(mic_pos))
    room.simulate()
    out = room.mic_array.signals[0][:len(signal)]
    if len(out) < len(signal):
        out = np.pad(out, (0, len(signal) - len(out)))
    return out.astype(np.float32)


def apply_reverb_to_pair(sig0, sig1, t60, pair_seed):
    """Independent RIR per speaker: same room, two distinct source positions
    (physically: two people at different spots in the same room), one mic."""
    rng = np.random.default_rng(pair_seed)
    room_dim = (6.0 + rng.uniform(-0.5, 0.5), 5.0 + rng.uniform(-0.5, 0.5), 3.0)
    mic_pos = (room_dim[0] / 2, room_dim[1] / 2, 1.5)
    src0_pos = (rng.uniform(1.0, room_dim[0] - 1.0), rng.uniform(1.0, room_dim[1] - 1.0), 1.5)
    src1_pos = (rng.uniform(1.0, room_dim[0] - 1.0), rng.uniform(1.0, room_dim[1] - 1.0), 1.5)
    rev0 = simulate_reverb(sig0, t60, src0_pos, mic_pos, room_dim)
    rev1 = simulate_reverb(sig1, t60, src1_pos, mic_pos, room_dim)
    # Renormalize each reverberant signal's peak back toward its pre-reverb peak
    # so reverb tail energy doesn't silently change relative speaker levels.
    for rev, orig in [(rev0, sig0), (rev1, sig1)]:
        orig_peak = np.max(np.abs(orig))
        rev_peak = np.max(np.abs(rev))
        if rev_peak > 1e-8:
            rev *= (orig_peak / rev_peak)
    return rev0, rev1


def minmax(values):
    values = np.asarray(values, dtype=np.float64)
    span = values.max() - values.min()
    if span > 1e-12:
        return (values - values.min()) / span
    return np.zeros_like(values)


def compute_cam(model, audio, speaker, index, device):
    input_audio = audio.clone().detach().to(device).float().requires_grad_(True)
    gradcam = GradCAM(model, SINGLE_LAYER, device)
    try:
        with torch.enable_grad():
            _ = model(input_audio)
            target = model.vad_logits[0, speaker, index]
            target.backward()
            activations = gradcam.hook.activations
            gradients = gradcam.hook.gradients
            weights = gradients.mean(dim=tuple(range(2, gradients.ndim)), keepdim=True)
            raw = torch.relu((weights * activations).sum(dim=1)).detach().cpu().numpy()[0]
            return raw
    finally:
        gradcam.hook.remove_hooks()


def compute_pair_mae(left, right, pair_idx, seed_offset=5000):
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


def silero_reference_mask(vad_model, clean_signal, num_frames):
    wav_t = torch.from_numpy(clean_signal.astype(np.float32))
    timestamps = get_speech_timestamps(wav_t, vad_model, sampling_rate=SAMPLE_RATE, return_seconds=False)
    mask = np.zeros(num_frames, dtype=bool)
    for seg in timestamps:
        start_frame = seg["start"] // HOP_LENGTH
        end_frame = min(num_frames, seg["end"] // HOP_LENGTH + 1)
        mask[start_frame:end_frame] = True
    return mask


def build_mixture_for_condition(sig0, sig1, condition, babble_track, pair_idx):
    """Returns the final (possibly degraded) mixture for a given condition name."""
    s0, s1 = sig0, sig1
    if condition.startswith("reverb") or condition.startswith("combo"):
        t60 = float(condition.split("t60_")[1].split("_")[0])
        s0, s1 = apply_reverb_to_pair(s0, s1, t60, pair_seed=9000 + pair_idx)
    mixture = normalize_audio(s0 + s1)
    if condition.startswith("noise") or condition.startswith("combo"):
        snr = float(condition.split("snr_")[1])
        mixture = add_noise_at_snr(mixture, babble_track, snr)
        mixture = normalize_audio(mixture)
    return mixture


def main():
    device = "cpu"
    config = json.loads((gradcam_root / "configs" / "config_with_vad.json").read_text())
    model = module_arch.SeparationModel(**config["arch"]["args"]).to(device).eval()
    checkpoint = torch.load(gradcam_root / "weights" / "model_with_vad.pth", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=True)

    print("[*] Loading Silero VAD reference model...")
    vad_model = load_silero_vad()

    pairs = load_all_pairs()
    used_names = set()
    for p1, p2, _ in pairs:
        used_names.add(p1.name)
        used_names.add(p2.name)
    unused_paths = leftover_unused_paths(used_names)
    print(f"[*] {len(pairs)} pairs, {len(unused_paths)} leftover unused utterances available for babble-noise proxy")

    rng_babble = np.random.default_rng(seed=42)
    babble_track = build_babble_track(unused_paths, rng_babble)

    conditions = (
        ["clean"]
        + [f"reverb_t60_{t}" for t in T60_VALUES]
        + [f"noise_snr_{s}" for s in SNR_VALUES]
        + ["combo_t60_0.4_snr_5"]
    )
    num_frames = num_frames_for_length(TARGET_LENGTH)

    all_rows = {cond: [] for cond in conditions}
    for idx, (p1, p2, tag) in enumerate(pairs):
        sig0 = load_and_pad(p1)
        sig1 = load_and_pad(p2)

        # Reference VAD masks always computed on the ORIGINAL clean sources.
        ref_masks = [silero_reference_mask(vad_model, sig0, num_frames), silero_reference_mask(vad_model, sig1, num_frames)]

        for condition in conditions:
            mixture = build_mixture_for_condition(sig0, sig1, condition, babble_track, idx)
            audio = torch.from_numpy(mixture).unsqueeze(0)

            with torch.no_grad():
                output = model(audio.to(device))
                vad_logits = model.vad_logits.detach().cpu().numpy()[0]

            cams = []
            f1s = []
            for speaker in range(2):
                ref_mask = ref_masks[speaker]
                vad_idx = int(np.argmax(np.abs(vad_logits[speaker])))
                raw_cam = compute_cam(model, audio, speaker, vad_idx, device)
                cams.append(raw_cam)
                if ref_mask.sum() > 0 and ref_mask.sum() < len(ref_mask):
                    cam_norm = minmax(raw_cam)
                    cam_norm = cam_norm[:num_frames] if len(cam_norm) >= num_frames else np.pad(cam_norm, (0, num_frames - len(cam_norm)))
                    _, best_f1 = best_f1_threshold(cam_norm, ref_mask)
                    f1s.append(best_f1)

            mae_stats = compute_pair_mae(cams[0], cams[1], idx, seed_offset=5000)
            row = {
                "pair_index": idx + 1, "tag": tag,
                "mae": mae_stats["ind_mae"], "random_mae": mae_stats["random_mae"],
                "cam_best_f1_mean": float(np.mean(f1s)) if f1s else float("nan"),
            }
            all_rows[condition].append(row)

        if (idx + 1) % 5 == 0:
            print(f"  Processed {idx+1}/{len(pairs)} pairs across all {len(conditions)} conditions...")

    output_dir = gradcam_root / "results" / "robustness"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {"num_pairs": len(pairs), "target_layer": SINGLE_LAYER, "conditions": {}}
    for condition in conditions:
        rows = all_rows[condition]
        real = [r["mae"] for r in rows]
        rand = [r["random_mae"] for r in rows]
        wstats = compute_wilcoxon_stats(real, rand)
        f1s = [r["cam_best_f1_mean"] for r in rows if not np.isnan(r["cam_best_f1_mean"])]
        summary["conditions"][condition] = {
            "mae_mean": float(np.mean(real)), "mae_std": float(np.std(real)),
            "random_mae_mean": float(np.mean(rand)), "random_mae_std": float(np.std(rand)),
            "separation_gap": float(np.mean(rand) - np.mean(real)),
            "wilcoxon_p": wstats["p_value"], "rank_biserial_r": wstats["rank_biserial_r"],
            "cam_best_f1_mean": float(np.mean(f1s)) if f1s else float("nan"),
            "cam_best_f1_std": float(np.std(f1s)) if f1s else float("nan"),
        }
        csv_path = output_dir / f"robustness_{condition}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary_path = output_dir / "robustness_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 100)
    print("ROBUSTNESS TO NOISE AND REVERBERATION - SUMMARY")
    print("=" * 100)
    print(f"{'Condition':<22}{'MAE':>10}{'RandMAE':>10}{'Gap':>10}{'p-value':>12}{'r':>8}{'CAM F1':>10}")
    for condition in conditions:
        s = summary["conditions"][condition]
        print(f"{condition:<22}{s['mae_mean']:>10.4f}{s['random_mae_mean']:>10.4f}{s['separation_gap']:>10.4f}"
              f"{s['wilcoxon_p']:>12.2e}{s['rank_biserial_r']:>8.3f}{s['cam_best_f1_mean']:>10.4f}")
    print("=" * 100)
    print(f"[+] Saved per-condition CSVs and {summary_path}")


if __name__ == "__main__":
    main()
