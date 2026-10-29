"""Three causally linked ECG-feedback loops (independent of the PyQt runtime).

Reimplements the deterministic signal path in GeometricNeuronOriginReview:
  homeostatic edge_of_chaos -> checkerboard -> image-to-vector ->
  first four vector entries -> homeostatic edge_of_chaos.

ADDITION (not present in ecg.json): neighboring loops send a scalar signal
proportional to previous-tick output deviation from the setpoint. Every loop
has its own controller history and checkerboard. This is an ENGINEERED digital
network, not spontaneous physical neural relay.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
import json
from pathlib import Path
import math

import cv2
import numpy as np

IMG_SIZE = 256
SETPOINT = 0.6  # 0.5 original, modulated by ConstantSignalNode(value=1.0) * 0.1
TARGET_VARIANCE = 0.1
VARIANCE_WINDOW = 50


@lru_cache(maxsize=4096)
def spatial_samples(square_size: int, vector_dim: int = 256) -> tuple[float, float, float, float]:
    """Exactly reproduce the relevant CV2 resize/flatten/normalization operations.

    Only cache geometry, not history. ImageToVectorNode normalizes by max abs
    of the FULL resized flattened vector, even though only four coordinates are
    wired back to HomeostaticCouplerNode.
    """
    if square_size == 0:
        raise ValueError("checkerboard square_size is zero: original node would divide by zero")
    y, x = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    image = ((x // square_size + y // square_size) % 2).astype(np.float32)
    side = math.ceil(math.sqrt(vector_dim))
    vec = cv2.resize(image, (side, side), interpolation=cv2.INTER_AREA).ravel()[:vector_dim].copy()
    mx = float(np.max(np.abs(vec))) if len(vec) else 0.0
    if mx > 0:
        vec /= mx
    return tuple(float(v) for v in np.pad(vec[:4], (0, max(0, 4-len(vec))))[:4])


@dataclass
class ECGNode:
    output: float = SETPOINT
    history: deque = field(default_factory=lambda: deque(maxlen=200))
    regime: float = 0.0
    error: float = 0.0
    integral_state: float = 0.0
    measured_input: float = 0.0
    square_size: int = 0

    def copy(self) -> ECGNode:
        return ECGNode(self.output, deque(self.history, maxlen=200), self.regime,
                       self.error, self.integral_state, self.measured_input, self.square_size)

    def sample(self, dim: int) -> float:
        # Historical checkerboard node uses `value or .1` as its fallback.
        size_in = self.output if self.output else 0.1
        square_size = int(5 + size_in * 50)
        self.square_size = square_size
        return sum(spatial_samples(square_size, dim))

    def step(self, value: float) -> None:
        """Original _edge_of_chaos and step() update order, no PyQt/display."""
        self.measured_input = float(value)
        self.history.append(float(value))
        hist = list(self.history)
        variance = float(np.var(hist[-VARIANCE_WINDOW:])) if len(hist) > 10 else 0.0
        error = variance - TARGET_VARIANCE
        dev = value - SETPOINT
        if error > 0:
            # 2 sigmoid(z)-1 == tanh(z/2); original sharpness=3.
            out = SETPOINT + math.tanh(dev * 1.5) / 3.0
            self.regime = -1.0
        else:
            amplified = dev * (1.0 + abs(error) * 10.0)
            out = SETPOINT + math.tanh(amplified * 2.0)
            self.regime = 1.0
        self.error = error
        # Updated exactly as original; in edge_of_chaos this value does NOT
        # participate in output and should never be described as the integrator.
        if len(hist) > 20:
            self.integral_state = self.integral_state * 0.99 + float(np.mean(hist[-50:])) * 0.01
        self.output = out


@dataclass
class ThreeLoop:
    nodes: list[ECGNode] = field(default_factory=lambda: [ECGNode() for _ in range(3)])
    dim: int = 256
    link_gain: float = 0.2
    feed_ab: bool = True
    feed_bc: bool = True

    def copy(self) -> ThreeLoop:
        return ThreeLoop([n.copy() for n in self.nodes], self.dim, self.link_gain, self.feed_ab, self.feed_bc)

    def step(self, pulse_a: float = 0., pulse_b: float = 0., pulse_c: float = 0.,
             noise: tuple[float, float, float] = (0., 0., 0.)) -> tuple[float, float, float]:
        # Synchronous updates; all coupling uses previous-tick state.
        prev_out = [n.output for n in self.nodes]
        local_sums = [n.sample(self.dim) for n in self.nodes]
        values = [local_sums[0] + pulse_a,
                  local_sums[1] + (self.link_gain * (prev_out[0]-SETPOINT) if self.feed_ab else 0.) + pulse_b,
                  local_sums[2] + (self.link_gain * (prev_out[1]-SETPOINT) if self.feed_bc else 0.) + pulse_c]
        for n, value, eta in zip(self.nodes, values, noise):
            n.step(value + eta)
        return tuple(n.output for n in self.nodes)


def prepare(initial: float, pulse_amp: float, warm: int, duration: int, hold: int,
            dim: int = 256, gain: float = 0.2, noise_seq: np.ndarray | None = None) -> ThreeLoop:
    w = ThreeLoop(dim=dim, link_gain=gain)
    for n in w.nodes:
        n.output = initial
    t = 0
    for _ in range(warm):
        w.step(noise=tuple(noise_seq[t]) if noise_seq is not None else (0., 0., 0.)); t += 1
    for _ in range(duration):
        w.step(pulse_a=pulse_amp, noise=tuple(noise_seq[t]) if noise_seq is not None else (0., 0., 0.)); t += 1
    for _ in range(hold):
        w.step(noise=tuple(noise_seq[t]) if noise_seq is not None else (0., 0., 0.)); t += 1
    return w


def assay(initial: float, pulse_amp: float, warm: int, duration: int, hold: int,
          probe: float, steps: int, dim: int = 256, gain: float = 0.2,
          noise: float = 0., seed: int = 42) -> dict:
    """Two interventions; both explicitly distinguish ordinary transfer from memory.

    Natural: stimulate A, then disconnect A->B and compare B/C to sham. This
      includes any immediate B/C output and history differences.

    Memory graft: start from same sham reference state in ALL nodes; replace
      ONLY B.history with that from the stimulated arm. Keep B's current output,
      its square geometry, its integrator, AND all of C's state identical.
      Disconnect A->B; same probe at B in all arms. The ablation arms preserve
      base history or turn B->C off, allowing attribution to B's memory.
    """
    noise_all = np.random.default_rng(seed).normal(0., noise, size=(warm+duration+hold+steps, 3))
    prep_len = warm+duration+hold
    base = prepare(initial, 0., warm, duration, hold, dim, gain, noise_all[:prep_len])
    trained = prepare(initial, pulse_amp, warm, duration, hold, dim, gain, noise_all[:prep_len])
    v_base = float(np.var(list(base.nodes[1].history)[-VARIANCE_WINDOW:]))
    v_trained = float(np.var(list(trained.nodes[1].history)[-VARIANCE_WINDOW:]))

    nat0, nat1 = base.copy(), trained.copy()
    nat0.feed_ab = nat1.feed_ab = False
    natural = [[], []]
    for t in range(steps):
        eta = tuple(noise_all[prep_len+t])
        natural[0].append(nat0.step(noise=eta))
        natural[1].append(nat1.step(noise=eta))

    ref, graft, link_cut = base.copy(), base.copy(), base.copy()
    for w in (ref, graft, link_cut):
        w.feed_ab = False
    # The current measured input is also matched: splice only older history.
    # Without this, the history tail would contradict the copied latest input.
    written_past = list(trained.nodes[1].history)
    written_past[-1] = base.nodes[1].history[-1]
    graft.nodes[1].history = deque(written_past, maxlen=200)
    link_cut.nodes[1].history = deque(written_past, maxlen=200)
    link_cut.feed_bc = False
    cut_ref = base.copy(); cut_ref.feed_ab = False; cut_ref.feed_bc = False
    histories = [[], [], [], []]
    # No-probe counterfactuals: required to distinguish spontaneous history
    # effects from a *changed response to the same later input*.
    no_probe_ref = base.copy(); no_probe_ref.feed_ab = False
    no_probe_graft = base.copy(); no_probe_graft.feed_ab = False
    no_probe_graft.nodes[1].history = deque(written_past, maxlen=200)
    no_probe_histories = [[], []]
    for t in range(steps):
        # Identical later probe at B; a fixed short pulse, no pulse to A or C.
        stimulus = probe if t < 3 else 0.0
        eta = tuple(noise_all[prep_len+t])
        for i, w in enumerate((ref, graft, cut_ref, link_cut)):
            histories[i].append(w.step(pulse_b=stimulus, noise=eta))
        no_probe_histories[0].append(no_probe_ref.step(noise=eta))
        no_probe_histories[1].append(no_probe_graft.step(noise=eta))

    no_probe_histories = np.asarray(no_probe_histories)
    natural = np.asarray(natural)
    histories = np.asarray(histories)
    base_trace, graft_trace, cut0_trace, cut1_trace = histories
    b_natural = natural[1,:,1] - natural[0,:,1]
    c_natural = natural[1,:,2] - natural[0,:,2]
    b_diff = graft_trace[:,1] - base_trace[:,1]
    c_diff = graft_trace[:,2] - base_trace[:,2]
    cut_diff = cut1_trace[:,2] - cut0_trace[:,2]
    no_probe_c_diff = no_probe_histories[1,:,2] - no_probe_histories[0,:,2]
    incremental_c = c_diff - no_probe_c_diff
    no_probe_b_diff = no_probe_histories[1,:,1] - no_probe_histories[0,:,1]
    incremental_b = b_diff - no_probe_b_diff
    return {
        'input': dict(initial=initial, pulse_amp=pulse_amp, warm=warm, duration=duration,
                      hold=hold, probe=probe, steps=steps, dim=dim, gain=gain,
                      noise=noise, seed=seed),
        'at_checkpoint': {'B_variance_sham':v_base,'B_variance_written':v_trained,
                          'B_current_output_sham':base.nodes[1].output,
                          'B_current_output_written':trained.nodes[1].output,
                          'B_history_l2':float(np.linalg.norm(np.asarray(base.nodes[1].history)-np.asarray(trained.nodes[1].history))),
                          'C_current_output_sham':base.nodes[2].output,
                          'C_current_output_written':trained.nodes[2].output},
        'natural': {'B_peak_delta':float(np.max(np.abs(b_natural))),
                    'C_peak_delta':float(np.max(np.abs(c_natural))),
                    'C_rms_delta':float(np.sqrt(np.mean(c_natural**2)))},
        'memory_graft': {'B_peak_delta':float(np.max(np.abs(b_diff))),
                         'C_peak_delta':float(np.max(np.abs(c_diff))),
                         'C_rms_delta':float(np.sqrt(np.mean(c_diff**2))),
                         'C_no_BC_peak_delta':float(np.max(np.abs(cut_diff))),
                         'C_no_probe_peak_delta':float(np.max(np.abs(no_probe_c_diff))),
                         'C_probe_interaction_peak':float(np.max(np.abs(incremental_c))),
                         'C_probe_interaction_rms':float(np.sqrt(np.mean(incremental_c**2))),
                         'B_probe_interaction_peak':float(np.max(np.abs(incremental_b))),
                         'B_first_delta':float(b_diff[0]), 'C_first_delta':float(c_diff[0]),
                         'C_last_delta':float(c_diff[-1])},
        'traces': {'natural_C_sham':natural[0,:,2].tolist(),
                   'natural_C_pulse':natural[1,:,2].tolist(),
                   'graft_B_reference':base_trace[:,1].tolist(),
                   'graft_B_history':graft_trace[:,1].tolist(),
                   'graft_C_reference':base_trace[:,2].tolist(),
                   'graft_C_history':graft_trace[:,2].tolist(),
                   'graft_C_no_probe_sham':no_probe_histories[0,:,2].tolist(),
                   'graft_C_no_probe_written':no_probe_histories[1,:,2].tolist(),
                   'C_probe_interaction':incremental_c.tolist()}
    }


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument('--out', default='results/three_loop.json')
    p.add_argument('--initial',type=float,default=.6)
    p.add_argument('--pulse',type=float,default=.5)
    p.add_argument('--probe',type=float,default=.15)
    p.add_argument('--warm',type=int,default=140)
    p.add_argument('--duration',type=int,default=12)
    p.add_argument('--hold',type=int,default=8)
    p.add_argument('--steps',type=int,default=80)
    p.add_argument('--dim',type=int,default=256)
    p.add_argument('--gain',type=float,default=.2)
    p.add_argument('--noise',type=float,default=0.)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    out=assay(a.initial,a.pulse,a.warm,a.duration,a.hold,a.probe,a.steps,a.dim,a.gain,a.noise,a.seed)
    path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps({k:v for k,v in out.items() if k!='traces'},indent=2))

if __name__=='__main__':main()
