# Experiment 6 — causal slow-state relay and observer gating (2026-10-29)

This follows ThursdaySol's original [three-loop experiment](../README.md) **without changing it**. The original 50-tick variance controller produced a large but extremely fragile history-graft effect. Here we ask a different, constructive question: can a clearly specified **local, bounded, leaky state** make A→B→C transmission robust enough to decode from C's later externally visible output?

**Interpretation:** Yes, for this deliberately engineered model. No autonomous/biological or Klein–Gordon wave propagation is claimed.

## Run

From the repository root:

~~~bash
python -m pip install numpy opencv-python pytest
python -m experiments.slow_state_relay
python -m experiments.slow_state_relay --dim 1024 --out experiments/results/observer_1024.json
python -m experiments.slow_state_relay --noise 0.03 --out experiments/results/noise_0p03.json
python -m pytest -q test_three_loop.py experiments/test_slow_state_relay.py
~~~

The command writes complete numerical receipts in JSON. The default 256-D receipt is checked in at [results/slow_state_relay.json](results/slow_state_relay.json).

## Model: retained geometry, **new** memory law

Each of three loops still renders the original 256×256 checkerboard via integer square size, resizes it by OpenCV INTER_AREA, and reads the **first four** samples from the resolution-dependent vector. The original PyQt scheduler and variance controller are **not** used here.

A NEW local state m is added:

~~~text
q_i(t)       = mean(first four normalized spatial samples)
s_i(t)       = 1 + sensor_mod * (2*q_i(t) - 1)
drive_i(t)   = s_i(t) * (external_pulse_i + coupled_message_i) + independent_noise_i
m_i(t+1)     = clip(leak*m_i(t) + write_rate*tanh(drive_i/memory_scale), -1, +1)
out_i(t+1)   = 0.6 + output_gain*tanh((geometry_gain*(q_i-0.25)
                 + memory_read*m_i(t+1) + drive_i)/fast_scale)
message_i(t) = tanh(memory_emit*m_i(t)) [+ optional fast component]
~~~

Neighbors only receive one-tick-delayed messages. By default leak=.993, write_rate=.025, link_gain=.20, sensor_mod=.5, with no fast-emission term.

These are chosen engineering rules, not deductions from the original homeostat or from chain-neuron physics. In particular the positive result cannot retroactively validate the old 50-tick variance rule.

## Protocol and isolation

1. Warm three loops 80 ticks, with independent input noise.
2. Write ±0.30 into A for 12 ticks with only A→B enabled. Run 24 further ticks. B→C stays disconnected during writing, so C cannot directly see the original sender.
3. Cut A→B, erase A's residual state, and set B's current *visible output* to exactly 0.6, while preserving B's genuinely received hidden state. Then enable B→C for 14 ticks.
4. Cut B→C, wait 20 ticks, set C's visible output to exactly 0.6, and present the same +0.06 probe to C. Report C's output after the probe. A separate no-probe counterfactual uses identical future noise.
5. Ablate B's memory before transfer, cut B→C, or erase C's memory immediately before readout. Each trial is independently noisy across ± written states. The later matched-probe/no-probe pair shares noise **within** a trial.
6. Train a scalar output threshold on the first 20 positive and 20 negative preparations; then evaluate it on a separate 20+20 preparations. The decoder never receives the hidden memory as input.

The staged gates, resets, and deliberate initial write are *experimental interventions*. They guarantee an interpretable path but are not an emergent self-organizing chain.

## Main result (40 independent preparations per sign; noise SD 0.003)

| Condition | Held-out C output classification | Mean C hidden state for + / − writes | Probe interaction (+ minus −) |
|---|---:|---:|---:|
| Normal, 256-D | **100%** | +0.2431 / −0.2539 | −0.01028 |
| Erase B memory | 52.5% | +0.00094 / +0.00115 | ~+0.00002 |
| Cut B→C | 50.0% | +0.00068 / +0.00099 | ~+0.00002 |
| Erase C memory before probe | 50.0% | *pre-erasure* +0.2431 / −0.2539 | ~+0.00001 |

The direct B-memory knockout and severed B→C edge isolate the transfer path. Erasing C memory abolishes the readable downstream effect. The controller output is matched before the final probe.

**Noise boundary:** held-out C-output accuracy was 82.5% at noise SD 0.03 and 57.5% at SD 0.1. The relay is not universally reliable.

**Geometry boundary:** at vector dim 1024, C still retains a sign-decodable state, but its separation shrinks from 0.497 to about 0.301 and the *probe response interaction* shrinks from −0.01028 to −0.00028. With sensor_mod=0, the C-state separation is identical at 256 and 1024 (about 0.527): this isolates the **newly engineered geometry-sensitive gain** as the source of geometry dependence for state transmission. It is not evidence that the original ECG observation operator alone creates that relay.

## What this does and does not decide

The old three-loop experiment showed a fragile deterministic memory × probe switch. This new model gives a robust *constructive existence example*: the same checkerboard observer can participate in a reliable, local, slow-state causal relay after a new memory rule and sensor gate are specified.

This does **not** establish that the old ECG controller naturally evolves these rules, that near-criticality is necessary, that a biological action potential is modeled, or that the phase-memory/Klein–Gordon chains propagate information this way.

The useful next experiment is to **hold the new state law fixed and independently vary observation geometry, coupling range, and memory lifetime**, then ask whether any local adaptive law can maintain readout fidelity without hand-staged gates. Finally, transfer that test protocol—not these engineered positive outcomes—into the pumped-chain physics.
