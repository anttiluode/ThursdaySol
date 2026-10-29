"""Experiment 7: continuously coupled local homeostatic chain, no switched handoffs.

A, B and C always have the A->B and B->C connections active in the main runs.
The *engineered* local homeostat adjusts incoming gain from the node's own
activity (EMA of |memory|), without sender labels or nonlocal readout.
Probe readout uses a matched no-probe counterfactual with shared *future* noise.
Source/receiver memory knockouts and removed B->C are separate causal controls.
Not the original ECG variance controller and not physical wave propagation.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal
import numpy as np
from experiments.slow_state_relay import REST, Params, Loop


@dataclass
class Config:
    p: Params = field(default_factory=Params)
    target_activity: float = .22
    gain_speed: float = .009
    avg_speed: float = .025
    min_gain: float = .4
    max_gain: float = 2.2
    warm_ticks: int = 80
    write_ticks: int = 12
    hold_ticks: int = 55
    pulse_amplitude: float = .30
    probe_amplitude: float = .06
    initial_gain: float = 1.


@dataclass
class AdaptiveLoop:
    inner: Loop = field(default_factory=Loop)
    gain: float = 1.
    avg_activity: float = 0.

    def copy(self):
        return AdaptiveLoop(self.inner.copy(), self.gain, self.avg_activity)

    def step(self, p: Params, local_input: float, pulse: float,
             noise: float, cfg: Config, adapt: bool):
        # The gain is local and acts only on the incoming connection.
        self.inner.update(p, self.gain * local_input, pulse, noise)
        self.avg_activity = ((1-cfg.avg_speed)*self.avg_activity +
                             cfg.avg_speed * abs(self.inner.memory))
        if adapt:
            # Bounded proportional homeostasis based ONLY on local activity.
            self.gain = float(np.clip(self.gain + cfg.gain_speed *
                 (cfg.target_activity - self.avg_activity), cfg.min_gain, cfg.max_gain))


@dataclass
class ContinuousChain:
    cfg: Config = field(default_factory=Config)
    cells: list[AdaptiveLoop] = field(default_factory=lambda:[AdaptiveLoop() for _ in range(3)])

    def copy(self):
        return ContinuousChain(self.cfg, [c.copy() for c in self.cells])

    def step(self, pulse_a: float = 0., pulse_c: float = 0.,
             noise: tuple[float,float,float] = (0.,0.,0.),
             adapt: bool = True, cut_bc: bool = False, erase_b: bool = False):
        p = self.cfg.p
        # Synchronous propagation: neighbors receive PREVIOUS-TICK emissions.
        old_message = [c.inner.encode(p) for c in self.cells]
        incoming = (0.,p.link_gain*old_message[0],
                    0. if cut_bc else p.link_gain*old_message[1])
        for i, (c, signal, pulse, eta) in enumerate(zip(self.cells, incoming,
                                               (pulse_a,0.,pulse_c), noise)):
            c.step(p, signal, pulse, float(eta), self.cfg,
                   adapt=adapt and i>0)
            if i == 1 and erase_b:
                # Intervention: continuous local B-memory knockout.
                c.inner.memory = 0.


def trial(sign: int, seed: int, cfg: Config | None = None, *,
          adapt: bool = True, cut_bc: bool = False,
          erase_b: bool = False, erase_c: bool = False,
          erase_a_after_write: bool = False) -> dict:
    cfg=cfg or Config()
    rng=np.random.default_rng(seed)
    ch=ContinuousChain(cfg)
    for _ in range(cfg.warm_ticks):
        ch.step(noise=tuple(rng.normal(0,cfg.p.noise_sd,3)),adapt=adapt,
                cut_bc=cut_bc,erase_b=erase_b)
    for _ in range(cfg.write_ticks):
        ch.step(pulse_a=cfg.pulse_amplitude*sign,
                noise=tuple(rng.normal(0,cfg.p.noise_sd,3)),adapt=adapt,
                cut_bc=cut_bc,erase_b=erase_b)
    if erase_a_after_write:
        # Source silence *only* as a diagnostic intervention, never a handoff:
        # neither edge is gated; reset only source state and fast output.
        ch.cells[0].inner.memory=0.
        ch.cells[0].inner.output=REST
    for _ in range(cfg.hold_ticks):
        ch.step(noise=tuple(rng.normal(0,cfg.p.noise_sd,3)),adapt=adapt,
                cut_bc=cut_bc,erase_b=erase_b)
    before=[c.inner.memory for c in ch.cells]
    gains=[c.gain for c in ch.cells]
    avg=[c.avg_activity for c in ch.cells]
    if erase_c:
        ch.cells[2].inner.memory=0.
    # Output is NOT manually matched/reset here, unlike Experiment 6.
    stimulated, no_probe=ch.copy(),ch.copy()
    shared=tuple(rng.normal(0,cfg.p.noise_sd,3))
    stimulated.step(pulse_c=cfg.probe_amplitude, noise=shared,
                    adapt=adapt,cut_bc=cut_bc,erase_b=erase_b)
    no_probe.step(noise=shared,adapt=adapt,cut_bc=cut_bc,erase_b=erase_b)
    # SECONDARY local-memory assay: take a snapshot of C after the entire
    # uninterrupted propagation. For the ONE-TICK READOUT ONLY, set its
    # present visual output to the same REST value in every preparation and
    # remove fresh B input. Hidden C memory is untouched except in the
    # explicit erase-C control. This tests retained state rather than the
    # continuing carrier from B. It is NOT an intervention during transfer.
    isolated = ch.copy()
    isolated.cells[2].inner.output = REST
    local_probe, local_sham = isolated.copy(), isolated.copy()
    local_probe.step(pulse_c=cfg.probe_amplitude,noise=shared,
                     adapt=adapt,cut_bc=True,erase_b=erase_b)
    local_sham.step(noise=shared,adapt=adapt,cut_bc=True,erase_b=erase_b)
    return dict(sign=sign,seed=seed, C_mem=before[2], B_mem=before[1],
                A_mem=before[0], gains=gains,avg_activity=avg,
                output=stimulated.cells[2].inner.output,
                no_probe=no_probe.cells[2].inner.output,
                increment=stimulated.cells[2].inner.output-no_probe.cells[2].inner.output,
                isolated_output=local_probe.cells[2].inner.output,
                isolated_increment=local_probe.cells[2].inner.output-local_sham.cells[2].inner.output,
                C_read_after_reset=erase_c)


def accuracy(plus: np.ndarray, minus: np.ndarray) -> dict:
    """Freeze one calibrated threshold/sign on first half of each condition.

    Separate positive/negative preparations have independent noise/seeds.
    """
    assert len(plus)==len(minus) and len(plus)>=4
    n=len(plus)//2
    mid=(plus[:n].mean()+minus[:n].mean())/2
    orient=1. if plus[:n].mean()>minus[:n].mean() else -1.
    preds_plus=orient*(plus[n:]-mid)
    preds_minus=orient*(minus[n:]-mid)
    return dict(heldout_accuracy=float((np.count_nonzero(preds_plus>0)+np.count_nonzero(preds_minus<0)+
                .5*np.count_nonzero(preds_plus==0)+.5*np.count_nonzero(preds_minus==0))/(len(plus[n:])*2)),
                threshold=float(mid),direction=int(orient),
                mean_plus=float(plus.mean()),mean_minus=float(minus.mean()),
                separation=float(plus.mean()-minus.mean()),
                std_plus=float(plus.std()),std_minus=float(minus.std()))


def evaluate(cfg: Config | None = None, seeds:int=40, *, run_conditions: tuple[str,...] | None=None) -> dict:
    cfg=cfg or Config()
    options={
      'adaptive': dict(adapt=True),
      'fixed_gain': dict(adapt=False),
      'erase_A_after_write': dict(adapt=True,erase_a_after_write=True),
      'erase_B_throughout': dict(adapt=True,erase_b=True),
      'cut_BC': dict(adapt=True,cut_bc=True),
      'erase_C_before_probe': dict(adapt=True,erase_c=True),
      'fixed_erase_A': dict(adapt=False,erase_a_after_write=True),
    }
    names=run_conditions or tuple(options)
    report={'config':asdict(cfg),'seeds_per_sign':seeds,'conditions':{}}
    for name in names:
        opts=options[name]
        pos=[trial(+1,i*2+1,cfg,**opts) for i in range(seeds)]
        neg=[trial(-1,i*2+2,cfg,**opts) for i in range(seeds)]
        measures={k:accuracy(np.array([x[k] for x in pos]),np.array([x[k] for x in neg]))
                  for k in ('output','increment','isolated_output','isolated_increment','C_mem','B_mem','A_mem')}
        gs=np.array([x['gains'] for x in pos+neg])
        report['conditions'][name]={'readouts':measures,
                                    'gains_mean':gs.mean(axis=0).tolist(),
                                    'gains_min':gs.min(axis=0).tolist(),
                                    'gains_max':gs.max(axis=0).tolist(),
                                    'nonzero_probe':float(np.mean([abs(x['increment']) for x in pos+neg])),
                                    'example_pos':{k:pos[0][k] for k in ('gains','A_mem','B_mem','C_mem')},
                                    'example_neg':{k:neg[0][k] for k in ('gains','A_mem','B_mem','C_mem')}}
    return report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seeds',type=int,default=40)
    ap.add_argument('--noise',type=float,default=.003)
    ap.add_argument('--hold',type=int,default=55)
    ap.add_argument('--dim',type=int,default=256)
    ap.add_argument('--out',default='experiments/results/continuous_adaptive.json')
    args=ap.parse_args()
    cfg=Config(p=Params(dim=args.dim,noise_sd=args.noise),hold_ticks=args.hold)
    result=evaluate(cfg,seeds=args.seeds)
    path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n')
    print('noise',args.noise,'hold',args.hold,'observer dim',args.dim)
    for name,c in result['conditions'].items():
        print(f"{name:22} output={c['readouts']['output']['heldout_accuracy']:.1%} "
              f"increment={c['readouts']['increment']['heldout_accuracy']:.1%} "
              f"isolated={c['readouts']['isolated_output']['heldout_accuracy']:.1%} "
              f"Cmem={c['readouts']['C_mem']['separation']:+.4f} "
              f"gains={tuple(round(x,3) for x in c['gains_mean'])}")

if __name__=='__main__': main()
