# ECG / MIT-BIH Terminology Reference — Phase B Visualization Block

---

## 1. Raw Waveform

**Word definition:**
A "waveform" is a graph showing how a signal's amplitude changes over time. "Raw" means unprocessed — straight from the sensor/ADC, no filtering, no normalization, no artifact removal.

**MIT-BIH context:**
`record.p_signal` is a NumPy array of shape `(n_samples, n_leads)` containing the **physical signal** — voltage in millivolts, already converted from raw ADC integers using the record's gain and baseline (`p` = "physical," as opposed to `d_signal`, which is raw **digital** ADC units before conversion). Each column corresponds to a lead (in your case: `MLII`, `V5`). Your plot of `signal_ch0` over the first 10 seconds (3600 samples at 360 Hz) is the raw MLII lead — literally the electrical activity of the heart over time, unfiltered.

**Theory / mechanism:**
The heart's electrical activity is captured by electrodes on the skin. This tiny voltage (millivolts) is amplified and passed through an **Analog-to-Digital Converter (ADC)**, which samples it at a fixed rate — here, **360 Hz**, meaning 360 voltage readings per second. Each reading becomes one row in `p_signal`. The waveform you see is the composite of the heart's depolarization/repolarization cycle (P wave → QRS complex → T wave) repeated once per heartbeat. Because it's raw, it still contains baseline wander (slow drift from breathing/movement), powerline interference (60 Hz hum), and muscle noise — none of which has been removed yet. That's why later DL pipelines almost always add a filtering/denoising step before this signal reaches a model.

---

## 2. Annotated Waveform

**Word definition:**
"Annotation" = a label attached to a specific point or region of data. An "annotated waveform" overlays those labels on top of the raw signal plot.

**MIT-BIH context:**
`wfdb.rdann('./dataset/100', 'atr')` reads the `.atr` file — MIT-BIH's expert-labeled annotation file. Each record was independently labeled by **two or more cardiologists**, with disagreements adjudicated — this is why MIT-BIH is considered a gold-standard benchmark despite being small and old. `annotation.sample` gives the **sample index** (not time — you convert via `/ fs`) where each labeled event occurs, and `annotation.symbol` gives the **type** of event (e.g., `N` for normal beat). Your code overlays a red dot + text label at each annotated point on top of the continuous ECG trace.

**Theory / mechanism:**
Annotations in MIT-BIH are placed at (or very near) the **R-peak** — the tallest spike in the QRS complex — because it's the most unambiguous, highest-SNR landmark in a heartbeat, making it the natural reference point for both timing (RR intervals) and classification (beat type). The label itself encodes the cardiologist's diagnostic judgment about *that specific beat's* electrophysiological origin (e.g., did it originate in the sinus node normally, or was it an ectopic/abnormal beat from elsewhere in the heart). This is exactly the ground truth your eventual classifier will be trained against.

---

## 3. Dual-Lead Comparison

**Word definition:**
A "lead" in ECG is not a physical wire — it's a specific *view* of the heart's electrical field, computed from the voltage difference between two or more electrode placements. "Dual-lead" = plotting two such views simultaneously for comparison.

**MIT-BIH context:**
Every MIT-BIH record has **two leads**, but which two varies by record — most commonly `MLII` (Modified Limb Lead II) and one of `V1`, `V2`, `V4`, or `V5` (precordial/chest leads). Your record 100 has `MLII` and `V5`. `MLII` is the "primary" lead used in almost all MIT-BIH-based research because it gives the clearest, most consistent QRS complex — it's essentially the de facto standard for beat-detection algorithms trained on this dataset.

**Theory / mechanism:**
Different leads "see" the heart's dipole (its net electrical vector) from different angles, so the same heartbeat can look quite different across leads — a P-wave that's clearly visible in `MLII` might be flat or inverted in `V5`, and vice versa. This is genuinely useful: some abnormalities are more visible in one lead than another (this is part of why multi-lead 12-lead ECGs exist clinically). For a DL project, it raises a design decision you'll need to make soon: do you train on a single lead (simpler, matches most published MIT-BIH baselines), or use both leads as separate input channels (potentially more information, but not all records share the same second lead, so you'd need to handle inconsistent channel availability across the 48 records).

---

## 4. RR-Interval Scatter Plot (Poincaré Plot)

**Word definition:**
"RR interval" = the time gap between two consecutive R-peaks (i.e., two consecutive heartbeats). A "Poincaré plot" is a specific type of scatter plot from nonlinear dynamics/chaos theory where you plot each value in a time series against the *next* value in that same series — here, RRₙ (x-axis) vs. RRₙ₊₁ (y-axis).

**MIT-BIH context:**
Your commented-out code computes `rr_intervals = np.diff(annotation.sample) / fs * 1000` — the time (in ms) between every pair of consecutive annotated beats — then plots `rr_n` against `rr_n_plus_1`. Each dot represents one heartbeat transition. The diagonal reference line (`x = y`) represents perfectly regular timing — points falling exactly on it would mean every RR interval was identical to the one before it.

**Theory / mechanism:**
This is a classic tool from **Heart Rate Variability (HRV)** analysis. A tight, cigar-shaped cloud hugging the diagonal indicates a very regular rhythm (low variability — could be healthy resting rhythm, or could indicate reduced autonomic responsiveness depending on context). A wide, scattered cloud indicates high beat-to-beat variability — could be normal (respiratory sinus arrhythmia) or pathological (atrial fibrillation produces a famously chaotic, "shotgun-blast" Poincaré plot with no discernible shape). Quantitatively, this plot is usually summarized by two ellipse-fit measures: **SD1** (dispersion perpendicular to the line of identity — short-term/beat-to-beat variability) and **SD2** (dispersion along the line of identity — long-term variability). For your project specifically: **⚠️ this is currently a bug waiting to happen** — `annotation.sample` includes *all* annotation types, not just beats (see the `+` symbol at sample 18 in your printed table, which is a non-beat rhythm-change marker). If you don't filter to beat-only symbols first, `np.diff` will silently compute a garbage "RR interval" spanning from that non-beat marker to the first real beat, poisoning your very first scatter point. Filter first, then compute RR intervals.

---

## 5. Heartbeat Annotation Symbols (MIT-BIH Full Reference)

MIT-BIH's `.atr` files use a fixed symbol vocabulary defined by the WFDB/PhysioNet annotation standard. They fall into two categories: **beat labels** (what you'll train a classifier on) and **non-beat labels** (rhythm/signal-quality markers you should filter out before beat-level analysis).

### Beat annotation symbols (the actual heartbeat classes)

| Symbol | Meaning | Notes |
|---|---|---|
| `N` | Normal beat | The vast majority of beats in MIT-BIH (~75%) |
| `L` | Left bundle branch block beat | Conduction abnormality |
| `R` | Right bundle branch block beat | Conduction abnormality |
| `A` | Atrial premature beat | Ectopic beat originating in the atria |
| `a` | Aberrated atrial premature beat | Atrial premature beat with abnormal conduction |
| `J` | Nodal (junctional) premature beat | Originates near the AV node |
| `S` | Supraventricular premature beat | Broad category, rare in MIT-BIH |
| `V` | Premature ventricular contraction (PVC) | Ectopic beat originating in the ventricles — clinically important |
| `F` | Fusion of ventricular and normal beat | A PVC and a normal beat overlapping |
| `e` | Atrial escape beat | "Escape" = a backup beat when the normal pacemaker is late |
| `j` | Nodal (junctional) escape beat | |
| `E` | Ventricular escape beat | |
| `/` | Paced beat | From an artificial pacemaker |
| `f` | Fusion of paced and normal beat | |
| `Q` | Unclassifiable beat | Cardiologists couldn't confidently categorize it |

This is exactly the label set behind the famous **AAMI EC57 grouping**, which most published MIT-BIH classifiers actually collapse into 5 superclasses for training: `N` (normal), `S` (supraventricular ectopic: A, a, J, S, e, j), `V` (ventricular ectopic: V, E), `F` (fusion: F), `Q` (unknown: /, f, Q). Worth deciding early whether you're doing 15-class or 5-class classification — it changes your whole label-mapping step.

### Non-beat annotation symbols (filter these out before beat analysis)

| Symbol | Meaning |
|---|---|
| `+` | Rhythm change marker (the actual rhythm type is stored separately in `annotation.aux_note`, not `symbol`) — **this is what you saw at sample 18** |
| `~` | Signal quality change (noise onset/offset) |
| `\|` | Isolated QRS-like artifact, not a real beat |
| `[` / `]` | Start / end of ventricular flutter or fibrillation |
| `!` | Ventricular flutter wave |
| `x` | Non-conducted P-wave (blocked atrial beat, no QRS follows) |
| `p`, `t`, `u` | Peak of P-wave, T-wave, U-wave (fiducial point markers, not beat labels) |
| `(` / `)` | Waveform onset / offset markers |
| `"` | Comment annotation |
| `?` | Learning/unclassified marker |

**Practical takeaway for your pipeline:** before you do *anything* beat-level (classification labels, RR intervals, segmentation windows), filter `annotation.symbol` down to just the beat-symbol set above. A clean helper looks roughly like:

```python
BEAT_SYMBOLS = {'N','L','R','A','a','J','S','V','F','e','j','E','/','f','Q'}
beat_mask = np.isin(annotation.symbol, list(BEAT_SYMBOLS))
beat_samples = annotation.sample[beat_mask]
beat_symbols = np.array(annotation.symbol)[beat_mask]
```

This one filter fixes the Poincaré-plot bug from section 4 *and* gives you a clean label source for the classifier later.
