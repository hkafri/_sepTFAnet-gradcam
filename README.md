# Sep-TFAnet-VAD Grad-CAM

Grad-CAM interpretability for [Sep-TFAnet-VAD](https://github.com/MordehayM/Sep-TFAnet-VAD), a PyTorch
speaker-separation/VAD network — adapted from image-based Grad-CAM and validated on real LibriSpeech audio.

## Attribution

- **Base network and weights**: [MordehayM/Sep-TFAnet-VAD](https://github.com/MordehayM/Sep-TFAnet-VAD). This repo
  references and adapts that architecture (`network/model/model.py`, with one addition — see
  [Methodology](#methodology)) but does **not** redistribute the original repository's code or checkpoints.
- **Paper**:
  > Moradi, M., Gannot, S. et al. "Single-microphone speaker separation and voice activity detection in noisy and
  > reverberant environments." *EURASIP Journal on Audio, Speech, and Music Processing* (2025).
- **License note**: as of writing, the `Sep-TFAnet-VAD` repository does not publish a `LICENSE` file. No permissive
  reuse rights are assumed. This repo only references/attributes that network's architecture and weights for
  research/interpretability purposes; it does not redistribute them. You must obtain the weights directly from the
  original repository and are responsible for complying with whatever terms the original author sets.

## Setup

### Environment

```bash
python -m venv venv
# Windows: venv\Scripts\activate | Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

### Weights

Download `model_with_vad.pth` from [MordehayM/Sep-TFAnet-VAD](https://github.com/MordehayM/Sep-TFAnet-VAD) and place
it at:

```
weights/model_with_vad.pth
```

(`config_with_vad.json` under `configs/` is already included since it's a small text file describing the
architecture, not a weight.)

### LibriSpeech samples

No LibriSpeech audio is committed to this repo (it's copyrighted). Fetch two real utterances from two different
speakers with:

```bash
python scripts/fetch_librispeech_samples.py
```

This streams a couple of examples from `openslr/librispeech_asr` (Hugging Face `datasets`, streaming mode) so you
don't need to download the full ~346MB `test-clean.tar.gz`. It saves 2 utterances per speaker (4 files total) under
`data/librispeech_samples/`, since the sampler needs one utterance per speaker for the mixture plus a second as a
spare/anchor.

**Windows note**: Hugging Face's default audio decoder (`torchcodec`) requires FFmpeg shared libraries that may not
be present on your machine, causing a DLL load failure. The fetch script works around this by requesting
`Audio(decode=False)` and decoding the raw FLAC bytes manually with `soundfile`, so no FFmpeg install is required.

If Hugging Face streaming isn't reachable, the script falls back to `torchaudio.datasets.LIBRISPEECH(..., download=True)`,
which pulls the full `test-clean` subset. It never falls back to synthetic audio.

## Usage

```bash
# 1. Fetch 40 distinct real speakers (80 utterances) for 20 pairs
python scripts/fetch_librispeech_samples.py

# 2. Run target layer selection (scores all 24 TCN conv1d blocks)
python scripts/select_target_layer.py

# 3. Run single mixture verification figure
python scripts/verify_librispeech_gradcam.py --librispeech-root data/librispeech_samples --device cpu

# 4. Run multi-pair evaluation and statistical validation (Selection Set: 20 pairs)
python scripts/evaluate_multi_pair_gradcam.py --librispeech-root data/librispeech_samples --num-pairs 20 --seed-offset 1000

# 5. Fetch disjoint held-out set (30 new speakers / 15 pairs from validation split)
python scripts/fetch_librispeech_holdout.py

# 6. Run selection-bias-free held-out evaluation (15 pairs, target layer TCN.TCN.9.conv1d fixed)
python scripts/evaluate_multi_pair_gradcam.py --librispeech-root data/librispeech_holdout --num-pairs 15 --seed-offset 2000 --csv-output results/librispeech_gradcam/holdout_results.csv --summary-output results/librispeech_gradcam/holdout_summary.json --plot-output results/librispeech_gradcam/holdout_paired_comparison_plot.png
```

This pipeline will:
1. Stream 40 distinct real speakers from LibriSpeech `test-clean` to form 20 non-overlapping selection/evaluation pairs.
2. Quantitatively score all 24 TCN conv1d candidate layers using a non-monotonic entropy penalty and select the optimal layer (`TCN.TCN.9.conv1d`).
3. Fetch a completely disjoint set of 30 new speakers (15 pairs) from LibriSpeech `validation` (dev-clean) with zero speaker overlap.
4. Evaluate Grad-CAM across both sets and report selection-set, selection-bias-free held-out, and pooled aggregate statistics ($N = 35$ total pairs).

## Methodology

Three adaptation and evaluation decisions mattered and are worth calling out explicitly:

### 1. Pre-sigmoid VAD logit instead of the post-sigmoid probability

The VAD head ends in a `Sigmoid()`. Backpropagating from the post-sigmoid probability saturates almost everywhere
the model is confident (values pinned near 0 or 1), which collapses the upstream gradient and produces a
near-degenerate Grad-CAM. `network/model/model.py` exposes `model.vad_logits` — the same `VAD` module's
`output_layer_vad` output *before* the sigmoid (`VAD.forward_logits`, called from `SeparationModel.forward()`) — as a
non-saturating Grad-CAM target, without changing the model's normal sigmoid-probability output.

### 2. MAE instead of max-difference for comparing CAMs

An earlier iteration reported "two speakers' CAMs differ with max diff = 0.9999" as evidence of speaker-specific
attention. That number is not trustworthy on its own: max-elementwise-difference reads close to its theoretical
ceiling (~1.0) for *any* two sufficiently sparse activation maps — including one real CAM compared against a
**pure random-noise** map of the same shape — because sparse maps almost always have an isolated peak where the
other map is near zero. That's a property of sparsity, not evidence of disagreement.

This repo instead reports **mean absolute difference (MAE)**, plus the max-diff for context, for both an
independently min-max-normalized comparison and a shared-scale comparison, always alongside a random-noise control:

```
VAD-logit CAM, speaker vs speaker: MAE = 0.256 ± 0.109 (max diff ≈ 1.000)
VAD-logit CAM, speaker vs random noise: MAE = 0.409 ± 0.057 (max diff ≈ 0.934)
```

The max-diff numbers alone would suggest the real-vs-real and real-vs-random comparisons are equally "different."
The MAE numbers show the real speaker-vs-speaker CAMs are noticeably more self-similar to each other than either is
to random noise.

### 3. Principled layer selection via non-monotonic entropy scoring

Rather than picking an arbitrary layer index or monotonically rewarding diffuse heatmaps, all 24 TCN `conv1d` candidate layers (`TCN.TCN.0.conv1d` through `TCN.TCN.23.conv1d`) were systematically evaluated across all $N=20$ LibriSpeech speaker pairs using three quantitative metrics:
- **Gradient signal strength**: mean absolute gradient at that layer ($\text{mean}(|G|)$, log-transformed & z-scored).
- **Representation richness**: variance of activations at that layer ($\text{var}(A)$, log-transformed & z-scored).
- **CAM quality (non-monotonic)**: quadratic/absolute penalty for deviation from an optimal mid-range target entropy of $0.65$ ($-\text{abs}(H_{\text{norm}}(\text{CAM}) - 0.65)$, z-scored). This penalizes both degenerate uninformative uniform maps (entropy $\approx 1.0$) and single-pixel spike artifacts (entropy $\approx 0.0$).

Each metric was z-scored across layers and averaged to produce a combined layer score.
- **Winning layer**: `TCN.TCN.9.conv1d` (Block 9 of 24) — Combined Score: **+0.8931**, Mean Gradient: $1.96 \times 10^{-3}$, Activation Variance: $2.91$, CAM Entropy: $0.762$.
- Top runners-up: `TCN.TCN.11.conv1d` (+0.6865) and `TCN.TCN.15.conv1d` (+0.6148).

A visual comparison of CAM heatmaps across block depths ([cam_visual_comparison.png](results/layer_selection/cam_visual_comparison.png)) confirms that earlier blocks (e.g. Block 6) produce overly diffuse, blob-like attention maps, while mid-to-late blocks (e.g. Block 9 and Block 15) produce well-localized, temporally discriminative feature activations.

Full layer rankings and bar charts across all 24 blocks are saved in [layer_scores.json](results/layer_selection/layer_scores.json) and [layer_scores.png](results/layer_selection/layer_scores.png).

### Multi-layer ensemble vs. single-layer (tested, not adopted)

A natural follow-up question: does combining CAMs from multiple depths beat the single winning layer? Using the
same entropy-corrected ranking above (no new arbitrary picks), the top-scoring layer was taken from each depth
third: **early** (blocks 0–7) → `TCN.TCN.0.conv1d` (+0.5247), **middle** (blocks 8–15) → `TCN.TCN.9.conv1d`
(+0.8931, confirmed to still be the existing single-layer winner rather than assumed), **late** (blocks 16–23) →
`TCN.TCN.19.conv1d` (+0.2306). Their activation temporal lengths were verified (not assumed) to already match at
$T=188$ for the standard 3.0s mixture, so no interpolation was needed. The ensemble CAM is the elementwise average
of the three layers' individually min-max-normalized CAMs.

Both methods were run through the existing MAE-vs-random-control + Wilcoxon validation on all $N=35$ pooled pairs
(20 selection-set + 15 held-out, already fetched — no new data for this comparison):

| Method | Target | Real MAE | Random-Control MAE | Separation Gap (random − real) | Wilcoxon $p$ | Rank-Biserial $r$ |
|---|---|---|---|---|---|---|
| Single (`TCN.TCN.9.conv1d`) | VAD-logit | $0.2571 \pm 0.0931$ | $0.4130 \pm 0.0533$ | **$0.1559$** | $1.38\times10^{-7}$ | $0.902$ |
| Ensemble (Blocks 0, 9, 19) | VAD-logit | $0.2564 \pm 0.0528$ | $0.3621 \pm 0.0336$ | $0.1057$ | $1.75\times10^{-10}$ | $0.994$ |
| Single (`TCN.TCN.9.conv1d`) | Waveform | $0.2411 \pm 0.0710$ | $0.4217 \pm 0.0554$ | **$0.1807$** | $1.11\times10^{-9}$ | $0.978$ |
| Ensemble (Blocks 0, 9, 19) | Waveform | $0.2569 \pm 0.0455$ | $0.3536 \pm 0.0347$ | $0.0967$ | $2.50\times10^{-9}$ | $0.968$ |

**Verdict: the ensemble does not clearly help, and is kept out.** Absolute real-vs-real MAE is nearly identical
between the two methods, and the ensemble does have lower pair-to-pair variance (a real, if minor, upside). But
the metric that actually matters here — the **separation gap** between real speaker-vs-speaker MAE and the
random-noise control — is narrower for the ensemble on both targets, not wider: averaging three layers smooths
the CAM and pulls the random-control baseline down with it, shrinking the very signal this whole validation is
built to detect. Per the project's stated decision rule, single-layer `TCN.TCN.9.conv1d` remains the default CAM
method used throughout the rest of this README (Parts 2–4 below), not the ensemble.

Full per-pair numbers, summary stats, and the comparison plot are saved in
[ensemble_vs_single_results.csv](results/ensemble_cam/ensemble_vs_single_results.csv),
[ensemble_vs_single_summary.json](results/ensemble_cam/ensemble_vs_single_summary.json), and
[ensemble_vs_single_plot.png](results/ensemble_cam/ensemble_vs_single_plot.png).

### Finalized Block 9 Example

These figures use the finalized `TCN.TCN.9.conv1d` target layer and the existing four-panel renderer: mixture spectrogram, VAD-logit CAM curve, waveform CAM curve, and waveform-CAM overlay with the VAD-logit line. Both targets come from the same real LibriSpeech mixture formed from `5639-40744-0000.flac` and `61-70968-0001.flac`.

![Speaker 0 example using TCN.TCN.9.conv1d](results/librispeech_gradcam/example_speaker0_block9.png)

![Speaker 1 example using TCN.TCN.9.conv1d](results/librispeech_gradcam/example_speaker1_block9.png)

## Results

Statistical validation across **selection set ($N = 20$ pairs)**, **disjoint held-out set ($N = 15$ pairs)**, and **pooled dataset ($N = 35$ total pairs)** using target layer `TCN.TCN.9.conv1d` and freshly drawn independent random control maps per pair:

### VAD-Logit CAM Comparison

| Subset | $N$ Pairs | Real Speaker-vs-Speaker MAE | Real-vs-Random Control MAE | Wilcoxon $W$ | $p$-value | Rank-Biserial $r$ | Selection Bias Risk |
|---|---|---|---|---|---|---|---|
| **Selection Set** | $20$ | $0.2564 \pm 0.1094$ | $0.4090 \pm 0.0571$ | $17.0$ | $3.95 \times 10^{-4}$ | $0.838$ | Mild (Layer chosen on this set) |
| **Held-Out Set** | **$15$** | **$0.2580 \pm 0.0652$** | **$0.4185 \pm 0.0592$** | **$1.0$** | **$1.22 \times 10^{-4}$** | **$0.983$** | **None (Zero speaker overlap)** |
| **Pooled Total** | **$35$** | **$0.2571 \pm 0.0931$** | **$0.4131 \pm 0.0582$** | **$10.0$** | **$1.38 \times 10^{-7}$** | **$0.902$** | Minimal |

### Waveform CAM Comparison

| Subset | $N$ Pairs | Real Speaker-vs-Speaker MAE | Real-vs-Random Control MAE | Wilcoxon $W$ | $p$-value | Rank-Biserial $r$ | Selection Bias Risk |
|---|---|---|---|---|---|---|---|
| **Selection Set** | $20$ | $0.2497 \pm 0.0747$ | $0.4132 \pm 0.0523$ | $5.0$ | $1.91 \times 10^{-5}$ | $0.952$ | Mild (Layer chosen on this set) |
| **Held-Out Set** | **$15$** | **$0.2295 \pm 0.0640$** | **$0.4422 \pm 0.0554$** | **$0.0$** | **$6.10 \times 10^{-5}$** | **$1.000$** | **None (Zero speaker overlap)** |
| **Pooled Total** | **$35$** | **$0.2411 \pm 0.0710$** | **$0.4256 \pm 0.0555$** | **$1.0$** | **$8.15 \times 10^{-10}$** | **$0.981$** | Minimal |

### Key Artifacts & Visualizations

- **Selection Set Plot ($N=20$)**: [paired_comparison_plot.png](results/librispeech_gradcam/paired_comparison_plot.png)
- **Held-Out Set Plot ($N=15$)**: [holdout_paired_comparison_plot.png](results/librispeech_gradcam/holdout_paired_comparison_plot.png)
- **Pooled Total Plot ($N=35$)**: [pooled_paired_comparison_plot.png](results/librispeech_gradcam/pooled_paired_comparison_plot.png)
- **CSV Data**: [multi_pair_results.csv](results/librispeech_gradcam/multi_pair_results.csv) (selection set), [holdout_results.csv](results/librispeech_gradcam/holdout_results.csv) (held-out set), [pooled_results.csv](results/librispeech_gradcam/pooled_results.csv) (pooled set).
- **Layer Selection Artifacts**: [layer_scores.json](results/layer_selection/layer_scores.json), [layer_scores.png](results/layer_selection/layer_scores.png), [cam_visual_comparison.png](results/layer_selection/cam_visual_comparison.png).

The held-out validation confirms that the attention sensitivity effect is completely genuine and not an artifact of layer selection bias: on unseen, non-overlapping speakers, real speaker-vs-speaker MAE remains low ($0.229–0.258$), statistically significantly lower ($p < 0.0001$) than random control MAE ($0.418–0.442$), with an effect size of $r \ge 0.983$.

## VAD ground-truth alignment (IoU / F1)

**Caveat up front**: "reference" labels here come from running the open-source [Silero VAD](https://github.com/snakers4/silero-vad)
model on each **clean, pre-mix** single-speaker source. This is a reference from another automatic VAD model, **not
hand-labeled ground truth** — Silero VAD can itself misjudge onsets/offsets, so these numbers measure agreement with
a second automatic system, not an absolute correctness ceiling.

Using the finalized single-layer CAM (`TCN.TCN.9.conv1d`, VAD-logit target), computed on the mixture, thresholded
against the Silero reference mask for that speaker, over all $70$ speaker instances (35 pooled pairs × 2 speakers):

| Threshold | CAM F1 | CAM IoU | Network's own VAD-probability F1 (ceiling) | Network's own VAD-probability IoU |
|---|---|---|---|---|
| $0.3$ (fixed) | $0.346 \pm 0.191$ | $0.227 \pm 0.158$ | $0.919 \pm 0.098$ | $0.863 \pm 0.141$ |
| $0.5$ (fixed) | $0.194 \pm 0.140$ | $0.115 \pm 0.101$ | $0.913 \pm 0.105$ | $0.855 \pm 0.148$ |
| $0.7$ (fixed) | $0.081 \pm 0.072$ | $0.044 \pm 0.044$ | $0.901 \pm 0.105$ | $0.833 \pm 0.147$ |
| **Best-F1 per instance** | **$0.522 \pm 0.217$** (avg. optimal threshold $\approx 0.06$) | $0.383 \pm 0.207$ | **$0.939 \pm 0.081$** (avg. optimal threshold $\approx 0.31$) | $0.893 \pm 0.121$ |

**Honest read**: at fixed, "reasonable-looking" thresholds (0.3–0.7), CAM F1 is quite low and drops sharply as the
threshold rises (0.346 → 0.081), because the min-max-normalized CAM is sparse — most values sit well below 0.3, with
attention concentrated in a few peaks. The per-instance best-F1 threshold confirms this: it averages around $0.06$,
far below where one would naively threshold a normalized heatmap. At that best threshold, CAM reaches
$F1 = 0.522 \pm 0.217$ — **about 55.6% of the network's own VAD-prediction F1 ceiling** ($0.522 / 0.939$). This is a
real, moderate alignment with actual speech activity — clearly above chance — but well short of the network's own
VAD head, and the wide standard deviation ($\pm0.217$) means this varies substantially pair to pair.

Full per-instance results and thresholds: [vad_alignment_results.csv](results/vad_alignment/vad_alignment_results.csv),
[vad_alignment_summary.json](results/vad_alignment/vad_alignment_summary.json).

## Robustness to noise and reverberation

All prior results above were validated on **clean** mixtures only. This section re-runs the same 35 pooled pairs
(no new data — the point here is condition comparison, not sample-size scaling) through 3 reverberant conditions,
3 noisy conditions, and 1 combined condition, using the finalized single-layer CAM (`TCN.TCN.9.conv1d`).

- **Reverberation**: simulated with `pyroomacoustics` (image-source method), $T_{60} \approx 0.2\text{s} / 0.4\text{s} / 0.6\text{s}$.
  Each speaker's clean source is convolved with **its own independently simulated RIR** (same room, two different
  source positions relative to the microphone — physically, two people never share an impulse response) before
  the two are mixed. Levels are renormalized post-convolution to prevent reverb tail energy from silently changing
  relative speaker loudness.
- **Noise**: a babble-noise proxy built by summing 8 **leftover, unused** LibriSpeech utterances from the pool
  already fetched (70 of the 140 downloaded utterances were never used as a pair's "primary" utterance) — no new
  corpus download. Added at $\text{SNR} \approx 10\text{dB} / 5\text{dB} / 0\text{dB}$. **This is a low-effort
  proxy, not a real ambient-noise corpus** — a dataset such as [WHAM!](http://wham.whisper.ai/) would be a
  meaningfully stronger version of this test and is a natural next step.
- The Silero VAD reference mask (for the F1 metric) is always computed from the **original clean** pre-mix source,
  since "when did this speaker actually talk" doesn't change because of added reverb/noise — only what the network
  receives changes.

| Condition | Real MAE | Random-Control MAE | Separation Gap | Wilcoxon $p$ | Rank-Biserial $r$ | CAM Best-F1 vs. Silero |
|---|---|---|---|---|---|---|
| **Clean (baseline)** | $0.257 \pm 0.093$ | $0.415 \pm 0.053$ | $0.158$ | $1.87\times10^{-7}$ | $0.895$ | $0.522 \pm 0.134$ |
| Reverb $T_{60}=0.2$s | $0.227 \pm 0.076$ | $0.407 \pm 0.053$ | $0.181$ | $8.15\times10^{-10}$ | $0.981$ | $0.551 \pm 0.183$ |
| Reverb $T_{60}=0.4$s | $0.219 \pm 0.077$ | $0.410 \pm 0.054$ | $0.191$ | $5.82\times10^{-11}$ | $1.000$ | $0.509 \pm 0.158$ |
| Reverb $T_{60}=0.6$s | $0.231 \pm 0.088$ | $0.399 \pm 0.046$ | $0.167$ | $1.46\times10^{-9}$ | $0.975$ | $0.522 \pm 0.177$ |
| Noise $\text{SNR}=10$dB | $0.213 \pm 0.082$ | $0.417 \pm 0.053$ | $0.204$ | $2.91\times10^{-10}$ | $0.990$ | $0.509 \pm 0.176$ |
| Noise $\text{SNR}=5$dB | $0.216 \pm 0.105$ | $0.410 \pm 0.057$ | $0.194$ | $2.50\times10^{-9}$ | $0.968$ | $0.466 \pm 0.194$ |
| Noise $\text{SNR}=0$dB | $0.207 \pm 0.067$ | $0.422 \pm 0.041$ | $0.214$ | $1.16\times10^{-10}$ | $0.997$ | $0.470 \pm 0.132$ |
| Reverb $0.4$s + Noise $5$dB | $0.204 \pm 0.077$ | $0.407 \pm 0.046$ | $0.204$ | $2.91\times10^{-10}$ | $0.990$ | $0.454 \pm 0.154$ |

**Honest read — this did not come out the way a "robustness degrades under noise" narrative would predict:**

- **The MAE-vs-random-control separation** (the core speaker-discriminability signal from the rest of this README)
  **does not degrade** under any tested reverb or noise condition — if anything, the separation gap is *slightly
  larger* under noise ($0.194$–$0.214$ vs. $0.158$ clean) and comparable under reverb ($0.167$–$0.191$). All
  conditions remain highly statistically significant ($p < 10^{-6}$, $r > 0.89$). We do not have a confident
  explanation for why noise slightly *widens* this particular gap rather than narrowing it, and we are not
  claiming this means the model "sees better" under noise — it may reflect how the random-noise-control baseline
  interacts with a slightly different CAM sparsity pattern under degraded input; this is flagged as an open
  question rather than smoothed over.
- **The VAD-alignment F1 metric (Part 2) does show mild degradation, specifically under noise**, not reverb:
  F1 drops from $0.522$ (clean) to $0.466$–$0.509$ under the three noise conditions (roughly a 3–11% relative
  drop), while reverb alone stays within noise of the clean baseline ($0.509$–$0.551$, i.e. flat or marginally
  higher, likely within the run-to-run variance given the $\pm0.13$–$0.18$ standard deviations). The combined
  reverb+noise condition shows the lowest F1 of all eight conditions ($0.454$).
- **Bottom line**: within the tested range, added noise measurably degrades how well the CAM's attention aligns
  with actual speech activity (Part 2's ground-truth-alignment metric), even though it does not degrade the
  model's ability to tell the two speakers' CAMs apart from each other (the MAE metric). Reverberation at these
  T60 levels shows no clear negative effect on either metric. This is a genuine, reported-as-is finding, not
  softened for a cleaner narrative.

Full per-condition CSVs and the summary are saved in [results/robustness/](results/robustness/) (one CSV per
condition, plus [robustness_summary.json](results/robustness/robustness_summary.json)).

## Limitations / Next Steps

- **Sample Size & Diversity**: While extended to $N = 35$ total pairs across 70 distinct speakers, evaluations remain focused on clean speech mixtures from LibriSpeech. Noisy or reverberant environments should be tested in future work.
- **Single Target Layer**: Multi-pair and held-out validations fix the target layer to `TCN.TCN.9.conv1d` (the layer selected via Part A's ablation). Multi-layer or ensemble-based activation maps were not evaluated.
- **CPU Execution**: Verification and statistical benchmarks were conducted on CPU (`--device cpu`). GPU execution is supported via `--device cuda`.
