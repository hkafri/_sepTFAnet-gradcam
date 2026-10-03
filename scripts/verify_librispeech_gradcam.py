"""Verify Grad-CAM on a real LibriSpeech mixture.

The script requires a local LibriSpeech subset. It never silently replaces
missing corpus audio with synthetic data.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

gradcam_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(gradcam_root))

import network.model as module_arch
from data.librispeech import SpeakerSampler, load_utterance
from gradcam import GradCAM


SAMPLE_RATE = 16000
TARGET_SECONDS = 3.0


def get_target_layer():
    scores_path = gradcam_root / "results" / "layer_selection" / "layer_scores.json"
    if scores_path.exists():
        data = json.loads(scores_path.read_text())
        return data.get("winning_layer", "TCN.TCN.6.conv1d")
    return "TCN.TCN.6.conv1d"


TARGET_LAYER = get_target_layer()


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


def spectrogram(audio):
    window = torch.hann_window(512)
    return torch.abs(torch.stft(audio, 512, 256, window=window, return_complex=True)).numpy()


def compute_cam(model, audio, target_kind, speaker, index, device):
    input_audio = audio.clone().detach().to(device).float().requires_grad_(True)
    gradcam = GradCAM(model, TARGET_LAYER, device)
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


def report_comparison(left, right, label):
    independent_left = minmax(left)
    independent_right = minmax(right)
    shared_scale = max(float(left.max()), float(right.max()), 1e-12)
    shared_left = np.maximum(left, 0) / shared_scale
    shared_right = np.maximum(right, 0) / shared_scale
    random_map = np.random.default_rng(0).random(left.shape)
    print(f"  {label}: metric = max absolute elementwise difference")
    print(f"    independent min-max: max={np.max(np.abs(independent_left - independent_right)):.6f}, MAE={np.mean(np.abs(independent_left - independent_right)):.6f}")
    print(f"    shared raw scale:    max={np.max(np.abs(shared_left - shared_right)):.6f}, MAE={np.mean(np.abs(shared_left - shared_right)):.6f}")
    print(f"    real-vs-random control (independent min-max): max={np.max(np.abs(independent_left - random_map)):.6f}, MAE={np.mean(np.abs(independent_left - random_map)):.6f}")


def save_figure(spec, vad_cam, waveform_cam, paths, target_speaker, output_path):
    time = np.linspace(0, TARGET_SECONDS, spec.shape[1])
    vad_plot = np.interp(time, np.linspace(0, TARGET_SECONDS, len(vad_cam)), minmax(vad_cam))
    waveform_plot = np.interp(time, np.linspace(0, TARGET_SECONDS, len(waveform_cam)), minmax(waveform_cam))
    log_spec = np.log10(spec + 1e-8)
    source_text = " | ".join(path.name for path in paths)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.6), constrained_layout=True)
    axes[0].imshow(log_spec, aspect="auto", origin="lower", cmap="viridis", extent=[0, TARGET_SECONDS, 0, spec.shape[0]])
    axes[0].set_title(f"Mixture spectrogram (separation target: Speaker {target_speaker})")
    axes[0].set_xlabel("Seconds")
    axes[0].set_ylabel("Frequency bin")
    axes[1].plot(time, vad_plot, color="darkorange")
    axes[1].set_title(f"VAD-logit CAM (Speaker {target_speaker})")
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("Seconds")
    axes[1].set_ylabel("Normalized importance")
    axes[2].plot(time, waveform_plot, color="crimson")
    axes[2].set_title(f"Waveform-target CAM (Speaker {target_speaker})")
    axes[2].set_ylim(0, 1)
    axes[2].set_xlabel("Seconds")
    axes[2].set_ylabel("Normalized importance")
    axes[3].imshow(log_spec, aspect="auto", origin="lower", cmap="gray", extent=[0, TARGET_SECONDS, 0, spec.shape[0]])
    axes[3].imshow(np.tile(waveform_plot, (spec.shape[0], 1)), aspect="auto", origin="lower", cmap="inferno", alpha=0.6, extent=[0, TARGET_SECONDS, 0, spec.shape[0]])
    axes[3].plot(time, vad_plot * spec.shape[0], color="cyan", linewidth=1, label="VAD-logit CAM curve")
    axes[3].set_title(f"Waveform-target CAM heatmap + VAD-logit curve (Speaker {target_speaker})")
    axes[3].set_xlabel("Seconds")
    axes[3].set_ylabel("Frequency bin")
    axes[3].legend(loc="upper right")
    fig.suptitle(f"Real LibriSpeech, target speaker {target_speaker}: {source_text}", fontsize=12)
    fig.text(0.5, -0.02,
             "\"Waveform-target CAM\" is a Grad-CAM saliency curve backpropagated from the separated-waveform output, "
             "not the audio signal itself. A moderate, imperfect correspondence between the VAD-logit CAM\n"
             "and speech activity is the expected, already-quantified result: CAM-vs-Silero-reference best F1 = 0.522 vs. "
             "the network's own predicted-VAD F1 = 0.939 on the same reference (Part 2).",
             ha="center", va="top", fontsize=9)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--librispeech-root", type=Path, required=True, help="LibriSpeech subset directory")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not args.librispeech_root.exists():
        raise FileNotFoundError(f"LibriSpeech root does not exist: {args.librispeech_root}")

    sampler = SpeakerSampler(str(args.librispeech_root), seed=0)
    speaker_ids = sampler.sample_two_speakers()
    paths = [sampler.sample_utterance(speaker_id) for speaker_id in speaker_ids]
    if any(path is None or not path.exists() for path in paths):
        raise FileNotFoundError("Could not select two existing LibriSpeech utterances")
    print("Real LibriSpeech sources:")
    for speaker_id, path in zip(speaker_ids, paths):
        print(f"  speaker={speaker_id}, utterance={path.stem}, file={path.resolve()}")

    config_path = gradcam_root / "configs" / "config_with_vad.json"
    checkpoint_path = gradcam_root / "weights" / "model_with_vad.pth"
    config = json.loads(config_path.read_text())
    model = module_arch.SeparationModel(**config["arch"]["args"]).to(args.device).eval()
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=True)

    audio, _ = prepare_mixture(paths)
    with torch.no_grad():
        output = model(audio.to(args.device))
        vad_logits = model.vad_logits.detach().cpu().numpy()[0]
        separated = output[0].detach().cpu().numpy()[0]
    spec = spectrogram(audio[0])
    output_dir = gradcam_root / "results" / "librispeech_gradcam" / "final"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Store filenames only (not absolute local paths) so this metadata is safe to commit/publish.
    metadata = {"source_filenames": [path.name for path in paths], "speaker_ids": speaker_ids, "target_layer": TARGET_LAYER}
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    cams = {"vad_logit": [], "waveform": []}
    for speaker in range(2):
        vad_index = int(np.argmax(np.abs(vad_logits[speaker])))
        waveform_index = int(np.argmax(np.abs(separated[speaker])))
        vad_cam, vad_target = compute_cam(model, audio, "vad_logit", speaker, vad_index, args.device)
        waveform_cam, waveform_target = compute_cam(model, audio, "waveform", speaker, waveform_index, args.device)
        cams["vad_logit"].append(vad_cam)
        cams["waveform"].append(waveform_cam)
        print(f"speaker {speaker}: VAD logit frame={vad_index}, value={vad_target:.6f}; waveform sample={waveform_index}, value={waveform_target:.6f}")
        save_figure(spec, vad_cam, waveform_cam, paths, speaker, output_dir / f"example_speaker{speaker}.png")

    print("\nNormalization and control checks:")
    report_comparison(cams["vad_logit"][0], cams["vad_logit"][1], "VAD-logit CAM speaker comparison")
    report_comparison(cams["waveform"][0], cams["waveform"][1], "Waveform-target CAM speaker comparison")
    print(f"\nSaved final figures and source metadata to {output_dir}")


if __name__ == "__main__":
    main()