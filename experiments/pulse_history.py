"""Experiment 8: does a stream of events change a receiver's later probe response?

Reuses ThursdaySol Experiment 7's *engineered* continuous A->B->C chain.
Links remain live throughout warmup/write/hold. ONLY the diagnostic one-step
readout disconnects a new B->C input and matches C's fast visual output.

The primary result is a causal difference in C's probe increment, NOT a
claim of biological action potentials, spontaneous learned memory, or
multi-symbol storage with unlimited capacity.

Run: python -m experiments.pulse_history
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from itertools import product
import json
from pathlib import Path

import numpy as np

from experiments.continuous_adaptive import Config, ContinuousChain
from experiments.slow_state_relay import Params, REST


@dataclass(frozen=True)
class Protocol:
    warm_ticks: int = 80
    pulse_ticks: int = 4
    gap_ticks: int = 8
    final_hold: int = 8
    pulse_size: float = .06
    probe_size: float = .06
    noise_sd: float = .003
    observer_dim: int = 256


def sample_stream(symbols: tuple[int, ...], seed: int, protocol: Protocol = Protocol(),
                  *, adaptive: bool = True, cut_bc: bool = False,
                  erase_b: bool = False, erase_c: bool = False,
                  last_only: bool = False, ignore_probe: bool = False) -> dict:
    """Natural uninterrupted transport; interventions are separate controls.

    Independently seeded noise in each preparation; final probe/sham pair shares
    noise within that preparation. C is isolated ONLY at a one-tick diagnostic
    readout, *after* all stream transmission, never during the main transport.
    """
    if not symbols or any(s not in (-1, 1) for s in symbols):
        raise ValueError('symbols must be a nonempty tuple of +1 or -1')
    p = Params(dim=protocol.observer_dim, noise_sd=protocol.noise_sd)
    chain = ContinuousChain(Config(p=p))
    rng = np.random.default_rng(seed)

    def tick(pulse_a: float = 0.) -> None:
        eta = tuple(rng.normal(0., protocol.noise_sd, size=3))
        chain.step(pulse_a=pulse_a, noise=eta, adapt=adaptive,
                   cut_bc=cut_bc, erase_b=erase_b)

    for _ in range(protocol.warm_ticks):
        tick()
    for k, symbol in enumerate(symbols):
        # last_only control: preserve timing and simulation length, omit prior
        # symbols. Labels on omitted prior symbols must then be undecodable.
        amp = protocol.pulse_size * symbol if (not last_only or k == len(symbols)-1) else 0.
        for _ in range(protocol.pulse_ticks):
            tick(amp)
        if k < len(symbols)-1:
            for _ in range(protocol.gap_ticks):
                tick()
    for _ in range(protocol.final_hold):
        tick()

    c_memory = chain.cells[2].inner.memory
    b_memory = chain.cells[1].inner.memory
    c_fast = chain.cells[2].inner.output
    if erase_c:
        chain.cells[2].inner.memory = 0.
    # One-step diagnostic probe, not part of transmission: same fast output,
    # no new B field, same noise in pulse and sham arms. Hidden memory is
    # preserved unless erase_c requests a targeted knockout.
    chain.cells[2].inner.output = REST
    probe, sham = chain.copy(), chain.copy()
    eta = tuple(rng.normal(0., protocol.noise_sd, size=3))
    probe.step(pulse_c=0. if ignore_probe else protocol.probe_size,
               noise=eta, adapt=adaptive, cut_bc=True, erase_b=erase_b)
    sham.step(noise=eta, adapt=adaptive, cut_bc=True, erase_b=erase_b)
    response = probe.cells[2].inner.output
    baseline = sham.cells[2].inner.output
    return {'symbols': symbols, 'seed': seed,
            'C_memory_pre_erase': float(c_memory),
            'B_memory': float(b_memory),
            'C_fast_before_assay': float(c_fast),
            'C_baseline': float(baseline),
            'C_probe_output': float(response),
            'C_probe_increment': float(response-baseline)}


def _binary_decoding(values: np.ndarray, labels: np.ndarray, ntrain: int) -> dict:
    assert set(labels) == {-1, 1}
    train_x, test_x = values[:ntrain], values[ntrain:]
    train_y, test_y = labels[:ntrain], labels[ntrain:]
    pos = float(np.mean(train_x[train_y == 1]))
    neg = float(np.mean(train_x[train_y == -1]))
    threshold = (pos+neg)/2.
    orientation = 1 if pos >= neg else -1
    pred = np.where((test_x-threshold)*orientation > 0, 1, -1)
    return {'accuracy': float(np.mean(pred == test_y)),
            'train_threshold': threshold,
            'direction': orientation,
            'mean_plus': float(np.mean(values[labels==1])),
            'mean_minus': float(np.mean(values[labels==-1]))}


def evaluate_stream(length: int = 2, per_word: int = 80,
                    protocol: Protocol = Protocol(), *, adaptive: bool = True,
                    control: str = 'normal') -> dict:
    """Enumerate all possible ± pulse histories, with independent trial noise.

    Calibration and test sets use separate seeds for all words, prechosen rather
    than fitted based on held-out results. Decoders are scalar, class-balanced.
    """
    if length < 2 or length > 6:
        raise ValueError('length must be 2..6')
    if per_word < 8 or per_word % 2:
        raise ValueError('per_word must be an even integer >=8')
    options={'normal':{}, 'cut_BC':{'cut_bc':True},
             'erase_B':{'erase_b':True}, 'erase_C':{'erase_c':True},
             'latest_only':{'last_only':True}, 'no_probe':{'ignore_probe':True}}
    if control not in options:
        raise ValueError(control)
    words = list(product((-1,1), repeat=length))
    ntrain=per_word//2
    prep={w:[sample_stream(w, seed=2317 + 100_003 * j + t,
                           protocol=protocol, adaptive=adaptive,
                           **options[control]) for t in range(per_word)]
          for j,w in enumerate(words)}
    output={'protocol':asdict(protocol),'length':length,
            'per_word':per_word,'adaptive':adaptive,'control':control,
            'train_per_word':ntrain,'test_per_word':per_word-ntrain,
            'means':{},'bit_decoding':{},'word_decoding':{},'context_contrast':{}}
    for w in words:
        key=''.join('+' if x==1 else '-' for x in w)
        output['means'][key]={k:float(np.mean([r[k] for r in prep[w]]))
                              for k in ('C_memory_pre_erase','C_probe_output','C_probe_increment')}

    for obs in ('C_probe_output','C_probe_increment','C_baseline','C_memory_pre_erase'):
        # Order all TRAIN records by word then TEST records by word: independent.
        tr=[(w,r) for w in words for r in prep[w][:ntrain]]
        te=[(w,r) for w in words for r in prep[w][ntrain:]]
        train_vals = np.array([r[obs] for w,r in tr])
        test_vals = np.array([r[obs] for w,r in te])
        train_words=[w for w,r in tr]
        test_words=[w for w,r in te]
        n=len(tr)
        result_bits=[]
        for bit in range(length):
            values = np.concatenate((train_vals,test_vals))
            labels = np.array([w[bit] for w in train_words+test_words])
            result_bits.append(_binary_decoding(values,labels,n)['accuracy'])
        # Single-scalar centroid classifier for complete word, calibrated on
        # training only. Uniform random baseline = 1/(2**length).
        centroids=np.array([np.mean([r[obs] for r in prep[w][:ntrain]]) for w in words])
        predictions=[words[int(np.argmin(np.abs(centroids-v)))] for v in test_vals]
        output['bit_decoding'][obs]=result_bits
        output['word_decoding'][obs]=float(np.mean([pred==word for pred,word in zip(predictions,test_words)]))

    # Explicit conditional intervention contrast: hold last symbol fixed at +1,
    # and all intervening symbols also fixed at +1. Compare first symbol -/+.
    low=(-1,)+(1,)*(length-1)
    high=(1,)*length
    for obs in ('C_probe_output','C_probe_increment','C_baseline'):
        lo=np.array([r[obs] for r in prep[low][ntrain:]])
        hi=np.array([r[obs] for r in prep[high][ntrain:]])
        output['context_contrast'][obs] = {'mean_plus_history_minus_minus_history':float(hi.mean()-lo.mean()),
                                          'sd_plus':float(hi.std()),'sd_minus':float(lo.std())}
    return output


def evaluate_paired_context(protocol: Protocol = Protocol(), n_pairs: int = 120,
                            *, adaptive: bool = True, control: str = 'normal') -> dict:
    """Causal history x next-probe interaction using COMMON-RANDOM-NUMBER twins.

    Histories (-,+) and (+,+) share exactly the same per-tick noise and the
    same most recent + pulse. Only the earlier pulse differs. The pairwise
    difference is an intervention estimate; it must not be confused with
    independent-noise decoding reliability from evaluate_stream().
    """
    options = {'normal': {}, 'cut_BC': {'cut_bc': True},
               'erase_B': {'erase_b': True}, 'erase_C': {'erase_c': True},
               'latest_only': {'last_only': True}, 'no_probe': {'ignore_probe': True}}
    if control not in options:
        raise ValueError(control)
    if n_pairs < 4:
        raise ValueError('n_pairs must be >=4')
    delta = []
    for i in range(n_pairs):
        seed = 810200 + 17*i
        low = sample_stream((-1, +1), seed, protocol, adaptive=adaptive, **options[control])
        high = sample_stream((+1, +1), seed, protocol, adaptive=adaptive, **options[control])
        delta.append([high[k] - low[k] for k in
                      ('C_probe_increment', 'C_probe_output',
                       'C_baseline', 'C_memory_pre_erase')])
    data = np.array(delta, dtype=float)
    values = {}
    for j,k in enumerate(('increment', 'probe_output', 'baseline', 'memory_pre_erase')):
        mean = float(data[:,j].mean())
        sem = float(data[:,j].std(ddof=1) / np.sqrt(n_pairs))
        values[k] = {'paired_difference_mean':mean,'standard_error':sem,
                     'normal_approx_ci95':[mean-1.96*sem, mean+1.96*sem]}
    values['increment']['fraction_negative'] = float(np.mean(data[:,0] < 0))
    return {'protocol':asdict(protocol),'n_pairs':n_pairs,
            'adaptive':adaptive,'control':control,
            'histories':['-+','++'],
            'latest_pulse_identical':True,
            'same_noise_within_pair':True,
            'differences':values}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--length', type=int, default=2)
    parser.add_argument('--per-word', type=int, default=80)
    parser.add_argument('--noise', type=float, default=.003)
    parser.add_argument('--adaptive', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--control',default='normal')
    parser.add_argument('--out',default='experiments/results/pulse_history.json')
    a=parser.parse_args()
    result=evaluate_stream(a.length,a.per_word,Protocol(noise_sd=a.noise),
                           adaptive=a.adaptive,control=a.control)
    path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('stream',a.length,'noise',a.noise,'control',a.control,'adaptive',a.adaptive)
    print('increment:',result['bit_decoding']['C_probe_increment'],
          'word:',result['word_decoding']['C_probe_increment'])
    print('output:',result['bit_decoding']['C_probe_output'],
          'word:',result['word_decoding']['C_probe_output'])
    print('history response contrast:',result['context_contrast']['C_probe_increment'])


if __name__ == '__main__':
    main()
