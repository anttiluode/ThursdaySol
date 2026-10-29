"""Engineered, noise-tested slow-state relay built on the ECG observer geometry.

This is NOT a faithful extension of the original Perception Lab/homeostat:
- Uses the original checkerboard's exact four CV2 area-averaged samples.
- Replaces the fragile 50-tick variance switch with a local bounded leaky trace.
- Adds explicit neighbor-to-neighbor scalar coupling and staged link gating.
- Does not model synapses, physical waves, or spontaneous learning.

Run from ThursdaySol's root: python -m experiments.slow_state_relay
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from three_loop import spatial_samples

REST = 0.6


@dataclass
class Params:
    dim: int = 256
    leak: float = .993
    write_rate: float = .025
    memory_scale: float = .1
    link_gain: float = .20
    memory_emit: float = 2.0
    fast_emit: float = 0.0
    memory_read: float = .18
    geometry_gain: float = .04
    sensor_mod: float = .5
    fast_scale: float = .20
    output_gain: float = .24
    noise_sd: float = .003


@dataclass
class Loop:
    output: float = REST
    memory: float = 0.0
    observed: float = 0.0

    def copy(self) -> 'Loop':
        return Loop(self.output, self.memory, self.observed)

    def encode(self, p: Params) -> float:
        return math.tanh(p.memory_emit * self.memory) + p.fast_emit * (self.output - REST)

    def update(self, p: Params, incoming: float, pulse: float, noise: float) -> None:
        # Observation is the first four samples of the *present* checkerboard,
        # generated with the original integer-quantized square-size rule.
        square_size = int(5 + (self.output if self.output else .1) * 50)
        q = sum(spatial_samples(square_size, p.dim)) / 4.0
        self.observed = q
        # The geometry of the four sampled pixels changes sensitivity to
        # incoming activity. This gate is an EXPLICIT engineered addition.
        sense_gain = 1.0 + p.sensor_mod * (2.0 * q - 1.0)
        drive = sense_gain * (incoming + pulse) + noise
        self.memory = max(-1., min(1., p.leak * self.memory +
                                   p.write_rate * math.tanh(drive / p.memory_scale)))
        membrane = p.geometry_gain * (q - .25) + p.memory_read * self.memory + drive
        self.output = REST + p.output_gain * math.tanh(membrane / p.fast_scale)


@dataclass
class Chain:
    p: Params = field(default_factory=Params)
    cells: list[Loop] = field(default_factory=lambda: [Loop(), Loop(), Loop()])
    ab: bool = True
    bc: bool = False

    def copy(self) -> 'Chain':
        return Chain(self.p, [cell.copy() for cell in self.cells], self.ab, self.bc)

    def step(self, pulse_a: float = 0., pulse_b: float = 0.,
             pulse_c: float = 0., noise: tuple[float, float, float] = (0., 0., 0.)) -> None:
        # A single synchronous tick. Links read *previous* cell states.
        signals = [c.encode(self.p) for c in self.cells]
        incoming = [0., self.p.link_gain * signals[0] if self.ab else 0.,
                    self.p.link_gain * signals[1] if self.bc else 0.]
        for cell, field, pulse, eta in zip(self.cells, incoming,
                                           (pulse_a, pulse_b, pulse_c), noise):
            cell.update(self.p, field, pulse, eta)


def prepare(sign: int, seed: int, p: Params, *, write_ticks: int = 12,
            hold_ticks: int = 24) -> tuple[Chain, np.random.Generator]:
    """Write into A while A->B is on, but B->C is OFF; then cut A->B.

    C is not exposed to A's input until the distinct transfer phase.
    """
    rng = np.random.default_rng(seed)
    chain = Chain(p=p)
    for _ in range(80):
        chain.step(noise=tuple(rng.normal(0, p.noise_sd, 3)))
    for _ in range(write_ticks):
        chain.step(pulse_a=.30 * sign, noise=tuple(rng.normal(0, p.noise_sd, 3)))
    for _ in range(hold_ticks):
        chain.step(noise=tuple(rng.normal(0, p.noise_sd, 3)))
    chain.ab = False
    # No signal from A after the checkpoint, including stored A memory.
    chain.cells[0].memory = 0.
    chain.cells[0].output = REST
    # Remove B's fast output as an alternative explanation for the transfer.
    chain.cells[1].output = REST
    return chain, rng


def trial(sign: int, seed: int, p: Params, *, erase_b: bool = False,
          cut_bc: bool = False, erase_c: bool = False,
          write_ticks: int = 12, hold_ticks: int = 24,
          transfer_ticks: int = 14, read_delay: int = 20) -> dict:
    chain, rng = prepare(sign, seed, p, write_ticks=write_ticks, hold_ticks=hold_ticks)
    b_at_checkpoint = chain.cells[1].memory
    if erase_b:
        chain.cells[1].memory = 0.
    chain.bc = not cut_bc
    for _ in range(transfer_ticks):
        chain.step(noise=tuple(rng.normal(0, p.noise_sd, 3)))
    chain.bc = False
    # C's own trace can remain after the upstream link is removed.
    for _ in range(read_delay):
        chain.step(noise=tuple(rng.normal(0, p.noise_sd, 3)))
    c_memory = chain.cells[2].memory
    if erase_c:
        chain.cells[2].memory = 0.
    # Match C's instantaneous observable membrane state exactly before probe.
    chain.cells[2].output = REST
    # Counterfactual twins share identical future noise for causal probe effect.
    probe_arm, sham_arm = chain.copy(), chain.copy()
    eta = tuple(rng.normal(0, p.noise_sd, 3))
    probe_arm.step(pulse_c=.06, noise=eta)
    sham_arm.step(noise=eta)
    return dict(sign=sign, seed=seed, b_memory=b_at_checkpoint,
                c_memory=c_memory, probe_output=probe_arm.cells[2].output,
                sham_output=sham_arm.cells[2].output,
                probe_increment=probe_arm.cells[2].output-sham_arm.cells[2].output,
                after_erase_c=erase_c, after_erase_b=erase_b, cut_bc=cut_bc,
                observed=probe_arm.cells[2].observed)


def evaluate(p: Params = Params(), seeds: int = 40) -> dict:
    conditions = {'normal': {}, 'erase_B_memory': {'erase_b': True},
                  'cut_BC': {'cut_bc': True}, 'erase_C_memory': {'erase_c': True}}
    result: dict = {'parameters': vars(p), 'seeds': seeds, 'conditions': {}}
    for name, kw in conditions.items():
        positive = [trial(+1, s*2+1, p, **kw) for s in range(seeds)]
        negative = [trial(-1, s*2+2, p, **kw) for s in range(seeds)]
        pm = np.array([x['c_memory'] for x in positive])
        nm = np.array([x['c_memory'] for x in negative])
        pa = np.array([x['probe_output'] for x in positive])
        na = np.array([x['probe_output'] for x in negative])
        pi = np.array([x['probe_increment'] for x in positive])
        ni = np.array([x['probe_increment'] for x in negative])
        # Decoder uses only the externally visible output after a common probe.
        # Calibrate a ONE-DIMENSIONAL threshold on the first half of independently
        # noisy preparations, freeze it, and test on the held-out second half.
        ntrain = max(1, seeds // 2)
        threshold = .5 * (float(pa[:ntrain].mean()) + float(na[:ntrain].mean()))
        direction = 1. if pa[:ntrain].mean() >= na[:ntrain].mean() else -1.
        observable_acc = float((np.sum(direction*(pa[ntrain:]-threshold)>0) +
                                np.sum(direction*(na[ntrain:]-threshold)<0) +
                                .5*np.sum(pa[ntrain:] == threshold) +
                                .5*np.sum(na[ntrain:] == threshold)) / (2*(seeds-ntrain)))
        # Diagnostic privileged decoder reads the hidden C memory directly.
        # Treat exact zero as a tie to avoid punishing a null control artificially.
        acc = (np.sum(pm > 0) + np.sum(nm < 0) +
               .5 * (np.sum(pm == 0) + np.sum(nm == 0))) / (2*seeds)
        result['conditions'][name] = {
            'sign_accuracy_C_memory': float(acc),
            'heldout_accuracy_C_probe_output': observable_acc,
            'output_decoder_threshold': threshold,
            'mean_C_memory_plus': float(pm.mean()),
            'mean_C_memory_minus': float(nm.mean()),
            'mean_probe_output_plus': float(pa.mean()),
            'mean_probe_output_minus': float(na.mean()),
            'mean_probe_increment_plus': float(pi.mean()),
            'mean_probe_increment_minus': float(ni.mean()),
            'probe_interaction': float(pi.mean()-ni.mean()),
            'b_memory_abs_mean': float(np.mean([abs(x['b_memory']) for x in positive+negative])),
            'c_memory_abs_mean': float(np.mean(np.abs(np.r_[pm,nm]))),
            'c_memory_plus_std': float(pm.std()),
            'c_memory_minus_std': float(nm.std()),
        }
    return result


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='experiments/results/slow_state_relay.json')
    ap.add_argument('--seeds', type=int, default=40)
    ap.add_argument('--dim', type=int, default=256)
    ap.add_argument('--noise', type=float, default=.003)
    args = ap.parse_args()
    p = Params(dim=args.dim, noise_sd=args.noise)
    result = evaluate(p, seeds=args.seeds)
    path = Path(args.out); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    for name, d in result['conditions'].items():
        print(f"{name:18} sign={d['sign_accuracy_C_memory']:.1%} output={d['heldout_accuracy_C_probe_output']:.1%} "
              f"C+={d['mean_C_memory_plus']:.4f} C-={d['mean_C_memory_minus']:.4f} "
              f"probe interaction={d['probe_interaction']:.5f}")


if __name__ == '__main__':
    main()
