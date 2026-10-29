# Experiment 7 — continuously connected locally adaptive loops

Date: 2026-10-29. Standalone follow-up to [Experiment 6](RESULTS.md).

**Question:** Can a transient pulse write a state into a 3-loop A → B → C chain when both connections are active for the entire run, and can the resulting C state change the response to a later identical probe? Does a purely local gain homeostat improve reliability relative to fixed gain?

**Verdict:** Causal multihop state transfer exists in this **engineered leaky-memory model** without staged handoffs. A local gain homeostat is **not necessary** for the low-noise result and does **not** consistently improve fidelity. Neither this experiment nor the original ECG graph demonstrates biological neurons or self-organized physical waves.

## Run from the ThursdaySol repository root

~~~bash
python -m pip install numpy opencv-python pytest
python -m experiments.continuous_adaptive --seeds 160 --noise 0.003 --out experiments/results/continuous_adaptive_003.json
python -m experiments.continuous_adaptive --seeds 160 --noise 0.03 --out experiments/results/continuous_adaptive_03.json
python -m experiments.continuous_adaptive --seeds 160 --noise 0.1 --out experiments/results/continuous_adaptive_1.json
python -m pytest -q test_three_loop.py experiments/test_slow_state_relay.py experiments/test_continuous_adaptive.py
~~~

The compact saved [summary CSV](results/continuous_adaptive_summary.csv) records the 160-preparation-per-sign sweep for 3 independent noise levels and 7 conditions. The command can generate full raw numerical summaries.

## Rules: what is preserved, what is added

**Preserved from Experiment 6:** Three separate modules, each with a checkerboard controlled by its previous output, a 256-D (or configurable) area-downsample observer reading the first four pixels, an explicit bounded local leaky memory m, and one-tick-delayed scalar neighbor emission tanh(2m). The original 50-tick ECG variance controller is **not** used. Both the leaky memory and scalar communication are **engineered rules**, not emergent properties of the old Perception Lab.

**Different from Experiment 6:** Both A→B and B→C are live for **all** warm, write, and hold ticks. No stage-dependent link toggling, matched-output resets, sender memory erasure, or transplantation happens in the *main transfer experiment*. Only a 12-tick input train at A is applied; then it stops.

**Additional local homeostat:** Each receiving loop B and C has bounded incoming gain g and a local exponentially averaged activity trace a. The next gain depends exclusively on the loop's own memory magnitude:

~~~text
a_i(t+1) = 0.975 * a_i(t) + 0.025 * abs(m_i(t+1))
g_i(t+1) = clip(g_i(t) + 0.009 * (0.22 - a_i(t+1)), 0.4, 2.2)
received drive_i = g_i(t) * incoming_neighbor_signal_i
~~~

Gain starts at 1.0. It is not given the input label, a privileged receiver threshold, or downstream success. The fixed-gain control uses g=1 for all ticks. Adaptation is not a learned synaptic weight and cannot be equated with biological criticality.

## Protocol and readout isolation

1. Independently warm up 80 ticks at each loop with noise.
2. With both links continuously on, write either +0.30 or −0.30 to A for 12 ticks.
3. Stop writing, leave both links connected, and wait 55 ticks. No artificial handoff or reset.
4. Record the ordinary C output under a common +0.06 probe, plus a matched no-probe counterfactual sharing future noise.
5. **Secondary one-tick readout assay, not a propagation handoff:** clone C after propagation, match its immediate output to 0.6, and block its *new* input from B **for this measurement only**. Keep C's hidden memory unchanged. Apply the identical +0.06 probe (and a matched no-probe twin). The isolated output tests C's locally retained state, not B's current carrier.
6. Controls: remove B→C throughout, erase B memory every tick, erase C memory only before readout, erase A immediately after its write train, disable adaptation, or write no pulse. Diagnostic erasures/link removal do not occur in the main run.
7. For each condition, train a scalar threshold and orientation on the first 80 positive and 80 negative preparations; freeze it and score 80+80 *independently noisy* held-out preparations. The classifier observes C's resulting output, never its private memory.

## Results: held-out isolated C readout (160 test preparations)

| Noise SD | Adaptive gain | Fixed gain | Continuous B erase | B→C removed | C erased at read |
|---|---:|---:|---:|---:|---:|
| 0.003 | **100%** | **100%** | 47.5% | 47.5% | 45.6% |
| 0.03 | 90.6% | 92.5% | 48.1% | 48.1% | 46.9% |
| 0.10 | 70.0% | 65.0% | 48.1% | 48.1% | 47.5% |

At noise SD 0.003, the C-state separation across opposite writes is 1.9796 for adaptive and 1.9071 for fixed gain. This is near saturation of the allowed m interval [-1,+1]; it is therefore a reliable **one-bit sign memory**, not a demonstration of an enormous usable state capacity.

**Probe response, not just probe output:** use difference of the isolated probe and same-noise sham. This change in response identifies the earlier write at 100% (noise .003), 72.5% (.03), and **55.6%** (.1) with adaptation. Thus under high noise, predicting C's held state (70%) is easier than predicting how strongly it responds to the next probe (55.6%, near chance).

**Causal evidence:** Continuously erasing B's memory or removing its connection to C abolishes the isolated output decoding. Erasing C's hidden memory immediately before isolated readout also destroys it. That last control is crucial because the ordinary in-situ C output remains highly decodable even after C memory erasure, simply because B continues broadcasting. The isolated assay removes that alternative.

**Sender-off diagnostic:** Erasing A's memory after the write, while keeping both connections physically on, still yields 100% isolated C-output decoding at noise .003 after a 55-tick hold. This is a deliberate knockout *control*, not part of normal propagation. After holds of 20, 55, and 150 ticks, isolated decoding was 98.75%, 100%, and 98.75% (160 test trials each), while the isolated **increment** decoded 94.4%, 93.8%, and 81.3%. The later response is not invariant.

**No-write sham:** Setting write amplitude to zero gives 48.75% held-out isolated output decoding across independent ± assigned labels (160 test trials). There is no reliable sign information when no sign has been written.

**Geometry:** At 40 preparations per sign, changing vector dimension from 256 to 1024 shrinks the adaptive C-memory separation from approximately 1.985 to 1.562 at noise .003. The dimension changes the local observation geometry in the already engineered sensor rule. This is not evidence of emergent dendrites.

## Interpretation and remaining negative result

This is a successful demonstration of **naturally timed, uninterrupted, causal propagation of an engineered slow state** in a digital, feed-forward network. There are no hand-staged live edges in the main propagation. The results are not evidence for a physical action potential, original ECG homeostat criticality, spontaneously learned links, or the pumped Klein–Gordon chain physics.

The **adaptation hypothesis remains inconclusive**: both adaptive and fixed networks are perfect at low noise, fixed is slightly better at medium noise, adaptive better at high noise. The high-noise advantage is exploratory because several noise regimes and geometries were inspected; it needs separately preregistered replication. Under high noise, the *conditional change in probe response* is near chance.

Next useful falsification: maintain an intermediate activity state without sign saturation, test random symbol streams rather than one long ± write, and measure information at C after both the source and the intermediate emissions have ceased, without allowing the diagnostic isolated readout to act as a stand-in for physical signal isolation.
