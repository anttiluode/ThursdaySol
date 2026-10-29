# Experiment 9 — source-only phase-memory return through a reciprocal physical field

**2026-10-29 · independent follow-up to ThursdaySol Experiments 5–8**

## Verdict

**A qualified, causal YES for an internal-state-mediated return effect; NO for a proven A→B→C→A directed relay.**

A phase kick applied to oscillator A changes the wave emitted by A. Two other pinned near-complement pairs B and C develop small, history-dependent changes in their **difference phase** D. At a later checkpoint, a controlled transplant of **only the B/C difference-phase mode** (keeping the entire field, A's own state, and B/C's sum phase identical) changes how A responds to an identical later pulse. If B/C are not allowed to emit into the field, this difference vanishes.

The effect is very small and changes sign across preparations: 12/12 deterministic seeds showed some nonzero difference, but 4 were positive and 8 negative. The **signed mean is NOT distinguishable from zero in this small sweep**. No independent-noise decoding or reliable multi-hop regeneration has been established.

**Crucial leakage correction:** an earlier attempt to excite A using an external field source near A also wrote B and C **when A's emission was disabled**. That was direct field leakage, not an A-emitted return. The strict main experiment replaces that with a *local A phase kick*, with NO external field source while the changed phase propagates.

## Files

- [physical_subset.py](physical_subset.py): standalone NumPy port of the original oscillator-phase and Klein–Gordon field equations from [ChainNeuronsWithPhaseSpike chains/world.py](https://github.com/anttiluode/ChainNeuronsWithPhaseSpike/blob/main/chains/world.py); original isolated self-field calibration; pinned monomers; no extra memory variables
- [assay.py](assay.py): write, wait, matched-history counterfactuals, later probe and statistics
- [test_assay.py](test_assay.py): causal A-emission, checkpoint matching, B/C emission and zero-write tests
- [direct_field_leakage.py](direct_field_leakage.py): reproduce the rejected external-source condition
- [results/native_return_12seeds.csv](results/native_return_12seeds.csv): measured per-seed physical phase and probe-increment differences (rounded to 8 decimals)

## Equations and scope

Pinned oscillators obey the same *functional* rules used in the original simulation:

~~~text
psi_next = (2 - gamma)*psi - (1 - gamma)*psi_prev
           + c2*laplacian(psi) - m2*psi + sum_i g_emit*cos(theta_i)*kernel_i

theta_i_next = theta_i + omega_i
               - g_phase*cos(2*Omega*t)*heard_i*sin(theta_i)/omega_i

heard_i = spatial kernel average of psi - calibrated own emission
~~~

Pair frequencies are A = .03 and X = .11 + delta, delta=.015; pump = .14 rad/tick, N=48, gamma=.01, c2=.3, m2=.02, g_emit=.1, g_phase=.02. The wave-grid boundaries are **periodic**, with three pair centers at (24,21), (28,26), (20,26).

**Reimplementation limitations:** This is NOT a direct import or test against the original Torch World class. It uses float64 NumPy and NumPy-seeded initial phases rather than Torch float32 and Torch RNG. Forces, movement, bonds and ligation were omitted because all six monomers are pinned and none of those dynamics can move a pinned monomer or form bonds in this protocol. The only extra mechanisms are **explicit causal interventions** (A phase kick, phase-mode transplants, and emission masks for controls); no artificial leaky memory or local homeostat was added.

**Topology limitation:** The field is reciprocal and isotropic. B and C can each hear A directly. This is a triangular closed interaction network, *not a directionally isolated* A→B→C→A route. We have not established that C was influenced via B or that the effect survives multi-hop transport.

## Predeclared comparison within the chosen protocol

1. Warm the three physical pairs for 3,000 ticks with no external forcing; fork the same full starting state.
2. Apply a single **+0.6 rad phase kick to A's first oscillator** in the written fork only. Leave both forks running for 800 ticks with no external field source. Only A's altered native emission can initially distinguish the two worlds.
3. Measure the B/C difference phases \(D_k=\theta_{A,k}-\theta_{X,k}\) and sum phases \(S_k=\theta_{A,k}+\theta_{X,k}-0.14t\). Construct counterfactual checkpoints by taking the sham world and transplanting **only** B and/or C's D, shifting the pair phases in opposite directions. All other state, including A's phases, psi, psi_prev, and B/C sum phases, is matched exactly.
4. Apply an identical +0.15 external **0.11 rad/step carrier probe** near A from readout ticks 100–179. Compare against a no-probe sham fork with identical initial state and no stochastic noise. At readout tick 950 measure the unwrapped change in A's pair sum phase.
5. Define \(R=\left(S_A^{probe}-S_A^{sham}\right)_{D\,transplant}-\left(S_A^{probe}-S_A^{sham}\right)_{baseline}\). This removes the resting-phase difference as opposed to simply measuring a changed A output.
6. Controls: transplant B D alone, C D alone, their S phases alone, both D and S; additionally turn B/C emission OFF in both matched readout arms; silence A's emission during the source-only write; or apply zero kick.

The *phase-mode transplant* is a controlled intervention, not a naturally achieved checkpoint or a measurement you could perform on a living neuron. The response is measured in **radians**; no numerical normalization to unit field energy was used. We did **not** compute physical free energy or thermodynamic work extraction, so this is not a test of the Nature Physics energetic claim.

## Results (12 deterministic initial seeds)

| B/C phase mode transplanted | Mean signed change in A's probe increment (rad) | Mean absolute change (rad) | Approx. 95% interval for signed mean |
|---|---:|---:|---:|
| D_B only | −0.000921 | 0.001416 | [−0.002379, +0.000537] |
| D_C only | −0.001526 | 0.002698 | [−0.004318, +0.001266] |
| **D_B and D_C** | **−0.002425** | **0.004097** | **[−0.006674, +0.001823]** |
| S_B and S_C (no D) | −0.000629 | 0.001145 | [−0.001848, +0.000589] |
| Both S and D | −0.003050 | 0.005246 | [−0.008527, +0.002427] |
| D_BC; B and C emissions disabled in both arms | **0.000000** | **0.000000** | [0,0] |

The largest absolute D_BC difference in this exploratory sweep was 0.018813 rad; some preparations changed in the opposite direction. Thus the D degrees of freedom are a **causal contributor to a future return response**, not a stable uniform gain or robust multi-bit relay. S can contribute too; D is not the unique hidden state.

**Source-only emission check:** With A emission disabled in *both* worlds during an A phase kick, B and C were exactly identical in **8/8** seeded comparisons. An A phase change cannot reach B/C via that route unless A emits into the shared wave field.

**Direct external-field leakage control:** In 10 seeds, the original 0.11-frequency field pulse near A wrote noticeable B/C D changes even with A's emission disabled. It is therefore invalid to interpret the external-field write as having traveled via A. See [direct_field_leakage.py](direct_field_leakage.py) and run it independently.

### Reproduce

From repository root:

~~~bash
python -m pip install numpy pytest
OPENBLAS_NUM_THREADS=1 python -m experiments.physical_return.assay --seeds 12
OPENBLAS_NUM_THREADS=1 python -m experiments.physical_return.direct_field_leakage --seeds 10
OPENBLAS_NUM_THREADS=1 python -m pytest -q experiments/physical_return/test_assay.py
~~~

No GPU needed. The 12-seed full counterfactual sweep may run for a minute or more on a CPU.

## Interpretation and next discriminating experiment

This establishes the **mechanistic possibility** of:
A's phase → A emission → B/C changed D → B/C emission → altered future response of A.

It does **not** demonstrate the stricter *ordered* loop A→B→C→A, a self-sustaining regenerative traveling event, large addressable memory, an energy battery, or noise-tolerant information propagation.

Next: port the same counterfactual **directly into the original Torch World class**, verify numerical consistency to a defined tolerance and then build directionally restricted physical paths (or a spatial waveguide with appropriate matched controls). Measure route-specific transfer and signal-to-noise versus path length and probe delay. The original T6/T7 negative spike-relay evidence remains in force.
