# Experiment 8 — pulse history changes the next response, but stream capacity is limited

Date: 2026-10-29. Builds on the **engineered** continuously connected A → B → C model in [Experiment 7](CONTINUOUS_RESULTS.md). Code: [pulse_history.py](pulse_history.py); tests: [test_pulse_history.py](test_pulse_history.py); numerical receipt: [pulse_history_summary.json](results/pulse_history_summary.json).

## The question

**Does a receiver react differently to the same new pulse because of an earlier encounter?**

Answer: **Yes, in this already-engineered scalar memory model.** The *change in C's response* to an identical later probe depends on a prior pulse delivered only at A and relayed through B. Local B and C memory are necessary. However, that does **not** make this a robust memory for arbitrary streams: recently arriving symbols are poorly distinguishable, 3-bit word recall is limited, and even modest independent noise destroys decoding.

This is **not** the original Perception Lab ECG controller, an emergent geometric neuron, a model of ion-channel electrophysiology, or pumped-chain physics.

## Exact setup

Three continuously connected loops, same local leaky trace and checkerboard/first-four-pixel observer as Experiments 6–7, with optional gain homeostasis. No new learning law has been introduced for Experiment 8.

- Warm up **80 ticks**.
- Deliver **2 or 3 independent ±0.06 symbols** to A, each lasting **4 ticks**, with **8 ticks** between symbols. Both A→B and B→C remain connected during warmup, all pulses, and the **8-tick** final hold.
- At the **one-tick diagnostic readout only**, match C's current visible output to **0.6**, suppress **new** incoming B→C activity, and give C a **+0.06 probe**. Compare its output with a counterfactual with **no probe**, using identical future noise within the pair.
- Primary measure is **C's probe increment** = (output after probe) − (output after no-probe). This distinguishes a change in the *effect of the same input* from a changed baseline output.
- Diagnostic controls separately erase B memory throughout, erase C memory only before readout, cut B→C throughout, omit earlier pulses while preserving timeline, omit the probe, or freeze adaptive gains.
- Independently noisy trials provide held-out decoding: a scalar threshold for each bit, or a nearest-centroid classifier for the entire pulse word. Training and test examples are independent; half the examples for each balanced word are held out.

The *readout isolation* is deliberately an intervention. It is **not** a self-isolating neuron or a naturally gated connection. The main propagation is uninterrupted.

## Causal matched-noise experiment

Compare histories **(−,+)** and **(+,+)**. The latest pulse is the same; everything about the timing and noise is the same except the sign of the earlier pulse. Within each history, C gets both an identical positive probe and a same-noise sham.

The context effect is the paired difference between the **probe increments** of the two histories. We used **120 independent pairs** per setting:

| Condition | Mean difference in C's probe increment | Approximate 95% interval |
|---|---:|---:|
| Adaptive gain, noise SD 0.003 | **−0.005139** | −0.005529 to −0.004750 |
| Fixed gain, noise SD 0.003 | **−0.003869** | −0.004118 to −0.003621 |
| Adaptive gain, noise SD 0.03 | −0.001713 | −0.002292 to −0.001134 |
| Fixed gain, noise SD 0.03 | −0.001492 | −0.001960 to −0.001025 |
| Erase B, cut B→C, erase C, omit earlier pulse, or omit probe | **0** | Exact zero in paired same-noise comparison |

In low noise **97.5%** of the 120 paired adaptive trials had a negative difference. At noise 0.03 that fell to **65.8%**, despite the remaining nonzero *average causal effect*. Intervals are elementary normal-approximation intervals over paired simulation differences, not a preregistered confirmatory analysis.

The probe increment difference is a few thousandths against an individual response increment around 0.065–0.070: observable, but small. The mechanism is the current model's saturating tanh response with a locally retained leaky trace, **not** a newly learned change in dendritic wiring.

## Independent-noise decoding: the limitation

| Readout from C's probe **increment** | First pulse | Second pulse | Third pulse | Complete word | Chance word |
|---|---:|---:|---:|---:|---:|
| 2 pulses, SD 0.003, adaptive | **81.25%** | 56.25% | — | **40.0%** | 25% |
| 2 pulses, SD 0.003, fixed gain | 80.63% | 55.63% | — | 40.0% | 25% |
| 2 pulses, SD 0.003, B-memory erased | 46.88% | 46.88% | — | 23.75% | 25% |
| 2 pulses, SD 0.003, C-memory erased | 43.13% | 46.88% | — | 23.13% | 25% |
| 2 pulses, SD 0.03, adaptive | 45.63% | 45.63% | — | 23.13% | 25% |
| 3 pulses, SD 0.003, adaptive | **85.42%** | 58.75% | 51.25% | **20.0%** | 12.5% |
| 3 pulses, SD 0.003, fixed gain | 86.25% | 60.42% | 50.42% | 25.0% | 12.5% |

Two-pulse results are based on all four balanced signed words, **80 preparations per word (40 train / 40 test)**, or 160 independent held-out observations. Three-pulse results use eight balanced words, **60 per word (30 train / 30 test)**, or 240 held-out observations.

**Main negative:** the memory responds disproportionately to the early symbol at these timescales. Later symbols have not produced comparably distinguishable downstream states at the chosen readout. There is no evidence of faithful 2- or 3-bit storage just because complete-word recognition exceeds chance.

**Control for recent-pulse-only:** retaining only the latest symbol reduces the first-symbol decoding to 47.5%, confirming the first-symbol effect is not an accidental input-label shortcut. But two-bit word accuracy is still only 40%, so the network is not a competent multi-bit sequence store.

**Adaptation not required:** fixed gain also shows the history × probe effect and has comparable held-out accuracy. One cannot attribute the causal effect specifically to gain homeostasis.

## Reproduce

From the ThursdaySol root (Python 3.10+ with numpy, opencv-python, pytest):

~~~bash
python -m experiments.pulse_history --length 2 --per-word 80 --noise 0.003
python -m experiments.pulse_history --length 3 --per-word 60 --noise 0.003 --out experiments/results/pulse3.json
python -m experiments.pulse_history --length 2 --per-word 80 --noise 0.03 --out experiments/results/pulse2_noise03.json
python -m pytest -q experiments/test_pulse_history.py
~~~

For the matched-noise causal experiment:

~~~python
from experiments.pulse_history import Protocol, evaluate_paired_context
print(evaluate_paired_context(Protocol(noise_sd=.003),n_pairs=120))
print(evaluate_paired_context(Protocol(noise_sd=.003),n_pairs=120,control='erase_C'))
~~~

The saved [JSON receipt](results/pulse_history_summary.json) reports the primary numerical results with full protocol values.

## What next?

Instead of interpreting the partial recall as a failure of memory itself, test whether changing **pulse spacing, local memory decay, geometry, and propagation delays** permits multiple recent symbols to remain independently recoverable under *independent* test noise. Freeze every parameter before confirmatory testing. Compare to a trivial leaky integrator with the same scalar state capacity. The geometry and homeostat must earn their complexity against that baseline.

Only after that would it be defensible to transfer the successful protocol into physical phase-memory chains and ask whether the required state law arises there.
