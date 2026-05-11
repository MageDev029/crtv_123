# Score-Improvement Discussion — TalkHead Subnet Miner

Compact summary of what moves the score on this subnet's executor rubric.
Captures levers that worked, dead ends we tested, and the architectural
ceiling we hit.

---

## Current performance

| Metric | This build (Wav2Lip + paste) | Top miner (miner_133, current logs) |
|---|---|---|
| final_score mean | **0.467** | 0.332 |
| final_score min | 0.448 | 0.290 |
| final_score max | 0.492 (one run hit 0.606) | 0.360 |
| std | 0.018 | 0.025 |
| quality_score | 0.498 | 0.370 |
| inference time | ~11 s | ~10 s |
| peak VRAM | ~0.9 GB | ~4.8 GB |
| 4-gate pass rate | 5 / 5 | (assumed) |
| **margin** | **1.41× top miner** | — |

Worst single-challenge run (0.448) beats top miner's best (0.360) by 0.088.
Wins every cycle.

---

## The scoring formula (one-page reference)

```
final = clamp01( quality × efficiency_factor )

quality = clamp01( gate × blended )                       # gate=1.0 when all 4 gates pass
blended = 0.35·identity + 0.35·lipsync + 0.15·video
        + 0.10·audio   + 0.05·temporal − penalty

efficiency_factor = exp( −0.15·time/30 − 0.10·vram/8 )    # × cap_penalty if over caps
```

**Hard caps (cap_penalty = 0.01× → final collapses if violated):**
- VRAM ≤ 10 GB
- inference time ≤ 30 s

**The 4 gates** (each failure multiplies `gate` and crashes quality):
| # | Gate | Threshold | Penalty |
|---|---|---|---|
| 1 | `face_detect_ratio` | ≥ 0.80 | `gate ×= 0.2` |
| 2 | `identity` | ≥ 0.35 | `gate ×= 0.1` |
| 3 | `sync_c` | ≥ 0.45 | `gate ×= 0.2` |
| 4 | `wer` | ≤ 0.60 | **`quality = 0`** (hard kill) |

---

## Where the score actually comes from

Sample 5-pair mean breakdown (final = 0.467):

```
component       value    weight   contribution    headroom
identity        0.676    ×0.35     0.236          +0.040 to max (0.35)
lipsync         0.626    ×0.35     0.219          +0.046 to max
video           0.439    ×0.15     0.066          +0.017 to max   ← weakest
audio           0.847    ×0.10     0.085          +0.001 (effectively maxed)
temporal        0.527    ×0.05     0.026          +0.024 to max
penalty         0.134      —      −0.134          +0.134 recoverable
                                  ─────
                       blended:    0.498
                       × eff:      ×0.929
                       final:      0.467
```

**Two weights matter most**: identity (35 %) and lipsync (35 %) together
control 70 % of the quality score. Optimize those first.

---

## Design choices that worked

### 1. Wav2Lip-GAN (not MuseTalk, not Wav2Lip-HD)

The executor measures lipsync as **cross-correlation between mouth-pixel-delta
and audio RMS envelope** ([lipsync.py:84-104](../talkhead-subnet/talkhead-executor/executor/scoring/lipsync.py#L84-L104)).
Wav2Lip is SyncNet-supervised on exactly that signal — it scores high here
**even though** the output is only 96×96.

| Model | Native res | Inference time | VRAM | Sync metric fit |
|---|---|---|---|---|
| **Wav2Lip-GAN** | 96×96 | ~11 s | 0.9 GB | ✓ SyncNet-trained |
| MuseTalk | 256×256 | ~15 s | 3.2 GB | ✗ different training objective |
| SadTalker | 512×512 | ~30 s | 10 GB | ✗ + cap risk |
| Diff2Lip / Hallo / EMO | 512×512 | 60-120 s | 16-24 GB | ✗ cap bust |

### 2. Lower-half paste-back (the key identity win)

Replace only the **mouth/chin region** (mid_y → bbox bottom) with Wav2Lip's
prediction. Upper face stays the original input image.

```
identity: full-face-replace ≈ 0.55  →  lower-half-paste ≈ 0.68  (+0.13)
```

Top miners that replace the whole face pay an ~0.20 identity tax for the
identity (35 % weight) component. We don't.

### 3. InsightFace `buffalo_l` for bbox detection

Same model the executor's scoring uses ([identity.py:27](../talkhead-subnet/talkhead-executor/executor/scoring/identity.py#L27)).
Matching detectors means **`face_detect_ratio` = 1.000 every challenge** —
no risk of hitting the 0.80 gate.

### 4. Audio-envelope-driven motion (not fixed sinusoids)

Generated head jitter from a smoothed combination of audio envelope and
bandlimited Gaussian noise. Non-periodic by construction.

| Metric | Fixed sinusoid | Audio-env + noise |
|---|---|---|
| `loop` penalty risk | high (corr of frame means) | ~0 |
| `motion_naturalness` | low | moderate |
| `temporal.smoothness` | low (sharp jerk) | high (bandlimited) |

### 5. Patch unsharp before seam-blend

Mild unsharp mask on the LANCZOS4-upscaled mouth patch (amount=0.35,
sigma=1.0). Recovers Laplacian variance lost to the 96→bbox upscale,
keeping the `blur` subscore of `video` from collapsing.

---

## Dead ends — things tested that did NOT help

Each was measured on the same 5-pair benchmark. **Don't waste time
re-exploring these on this rubric.**

| Change | Hypothesis | Result | Why it failed |
|---|---|---|---|
| Mel-window centering (shift back ~100 ms) | Reduces `sync_d` (mouth lag) | final 0.467 → 0.477 mean but **std 0.018 → 0.075** | Effective lag varies per input; fixed shift over/under-corrects |
| FFmpeg `-itsoffset -0.24` (advance audio) | Same fix, in mux | final 0.467 → 0.439 | Cuts 240 ms from audio start → WER rises 0.03 → 0.22 → `audio` drops 0.85 → 0.67 |
| Color-match patch to upper face | Lift identity (skin tone) | final 0.467 → 0.440 | +0.014 identity, but 10-15 s overhead → efficiency crashes |
| Shrunk paste region (only mouth band) | More original face → higher id | final 0.467 → 0.449 | Hurts lipsync more than it helps id; mouth chin motion matters |
| Global unsharp (full frame) | Uniform sharpness, lifts video | final 0.467 → 0.462, std up | Adds variance, marginal at best |
| GFPGAN restoration (`has_aligned=True`) | Detail recovery on Wav2Lip output | final 0.467 → 0.456 | Wav2Lip patch isn't actually face-aligned → restoration ineffective |
| GFPGAN restoration (proper, `has_aligned=False`) | Sharper face | **final 0.467 → 0.004** | 48 s inference → exceeds 30 s cap → cap_penalty 0.01× crash |
| MuseTalk + paste-back | Higher-res mouth + identity preserve | final 0.467 → 0.444 | Identity +0.045, but lipsync −0.043 + slower → net loss |
| MuseTalk batch_size 32 | Recover speed | **final 0.444 → 0.168** | VRAM 9.2 GB — occasional cap hits → catastrophic variance |

---

## Why we're stuck at ~0.47 — the architectural ceiling

The executor's lipsync metric is **specifically what SyncNet trains for**. Any
model not SyncNet-supervised loses on that 35 % weight, even if it has higher
visual quality:

```
                  Lipsync resolution    SyncNet-supervised?    Best on this rubric?
Wav2Lip-GAN       96×96                 yes                    ✓
MuseTalk          256×256               no (latent inpainting) ✗
SadTalker         512×512               no                     ✗
LatentSync        256×256               yes-ish (uses syncnet) maybe — not tested
DINet             256×256               yes-ish                untested
```

**No public open-source model exists that is**:
- SyncNet-supervised AND
- ≥ 256×256 output AND
- Under 30 s inference AND
- Under 10 GB VRAM

…which is what would be required to push past `final ≈ 0.55`.

---

## Paths to break the 0.50 ceiling (future work, none guaranteed)

In rough order of effort vs likely payoff:

### A. **LatentSync** (ByteDance, 2024)
- 256×256, diffusion-based, optimized for ~10-15 s inference
- Uses SyncNet loss in training → potentially scores well on this metric
- ~6 GB VRAM (under cap)
- **Untested by us.** Worth one experiment.

### B. **Train Wav2Lip variant at 256×256**
- The "correct" answer — Wav2Lip's architecture + SyncNet loss + higher-res training
- 1-2 weeks of ML work + dataset collection + GPU training
- Predicted final: 0.55-0.65

### C. **Wav2Lip + Real-ESRGAN post (mouth-only)**
- Run RealESRGAN-x2 on just the bbox region after seam-blend
- Adds ~2 s and ~1.5 GB VRAM
- Could lift `video` (currently 0.44) toward 0.55 without hurting identity
- Risk: SR can shift colors → identity drift

### D. **DINet**
- 256×256, mouth-only (same philosophy as Wav2Lip)
- Less mainstream, weights harder to find
- ~6 s inference if implemented right
- Worth scouting for a working repo

---

## Operational guidance

1. **Submit Wav2Lip+paste now.** The 1.41× margin is comfortable but not
   permanent — top miner improved +0.03 in the gap we've watched. Earn while
   the ceiling holds.
2. **Watch the gap monthly.** If top miner crosses 0.40, start prioritizing
   path A (LatentSync) or C (Wav2Lip + RealESRGAN).
3. **Don't repeat the dead ends.** They were measured, not guessed.
4. **Hard limits don't shift**: VRAM ≤ 10 GB, time ≤ 30 s, WER ≤ 0.60 — any
   change must fit in these regardless of how nice the quality gain looks.

---

## Key implementation invariants (don't accidentally break these)

If you iterate on [wav2lip_runner.py](wav2lip_runner.py), preserve:

- **`_mel_chunks` uses `np.tile(mel[:, -1:], (1, pad))`** — the `(1, pad)` is
  the bug fix for the original plan's flat-tile, easy to regress
- **InsightFace bbox** (not face_alignment / dlib) — matches the scorer
- **Mouth-only paste from `mid_y` to `sy2`** — not full-face, not shrunk
- **`_build_motion_trajectory` is non-periodic** — fixed sinusoids trip `loop`
- **Single `_unsharp` on the patch, not the full frame** — adds variance
- **AAC 192 kbps audio passthrough in final mux** — keeps WER ≈ 0
