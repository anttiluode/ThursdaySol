"""From repo root: python -m pytest -q experiments/test_continuous_adaptive.py"""
from dataclasses import replace
import pytest

from experiments.continuous_adaptive import Config, ContinuousChain, evaluate, trial
from experiments.slow_state_relay import Params


def test_normal_protocol_has_both_links_connected_every_tick():
    # Main chain has no ab/bc toggles and no stage-dependent handoff flags.
    model = ContinuousChain()
    assert not hasattr(model, 'ab') and not hasattr(model, 'bc')
    for i in range(20):
        model.step(pulse_a=0.3 if i < 4 else 0.)
    assert model.cells[1].inner.memory > 0
    assert model.cells[2].inner.memory > 0


def test_no_write_no_label_information():
    r = evaluate(Config(pulse_amplitude=0.), seeds=40,
                 run_conditions=('adaptive',))['conditions']['adaptive']
    assert abs(r['readouts']['C_mem']['separation']) < .1
    assert .25 < r['readouts']['isolated_output']['heldout_accuracy'] < .75


def test_live_chain_stores_readable_C_state_after_write():
    r = evaluate(seeds=40,run_conditions=('adaptive',))['conditions']['adaptive']
    assert r['readouts']['isolated_output']['heldout_accuracy'] >= .95
    assert r['readouts']['isolated_increment']['heldout_accuracy'] >= .9
    assert r['readouts']['C_mem']['separation'] > 1.8
    assert r['gains_mean'][1] > 1.05


def test_B_knockout_and_BC_cut_remove_heldout_local_C_information():
    rs = evaluate(seeds=40,run_conditions=('erase_B_throughout','cut_BC'))['conditions']
    for x in rs.values():
        assert .25 <= x['readouts']['isolated_output']['heldout_accuracy'] <= .75
        assert abs(x['readouts']['C_mem']['separation']) < .03


def test_erasing_C_memory_before_local_assay_destroys_readout():
    r=evaluate(seeds=40,run_conditions=('erase_C_before_probe',))['conditions']['erase_C_before_probe']
    assert .25 <= r['readouts']['isolated_output']['heldout_accuracy'] <= .75
    assert .25 <= r['readouts']['isolated_increment']['heldout_accuracy'] <= .75
    # C_mem is recorded pre-erase, so still records the original state.
    assert r['readouts']['C_mem']['separation'] > 1.8


def test_source_erasure_during_uninterrupted_coupling_still_allows_memory_relay():
    r=evaluate(seeds=40,run_conditions=('erase_A_after_write',))['conditions']['erase_A_after_write']
    assert r['readouts']['isolated_output']['heldout_accuracy'] >= .9
    assert r['readouts']['C_mem']['separation'] > .75


def test_adaptation_not_necessary_in_low_noise():
    r=evaluate(seeds=40,run_conditions=('adaptive','fixed_gain'))['conditions']
    assert r['adaptive']['readouts']['isolated_output']['heldout_accuracy'] == 1.
    assert r['fixed_gain']['readouts']['isolated_output']['heldout_accuracy'] == 1.
    assert r['fixed_gain']['gains_mean']==[1.,1.,1.]
    assert r['adaptive']['gains_mean'][1] != 1.


def test_high_noise_degrades_heldout_readout():
    r=evaluate(Config(p=Params(noise_sd=.1)), seeds=40,run_conditions=('adaptive',))
    assert r['conditions']['adaptive']['readouts']['isolated_output']['heldout_accuracy'] < .9


def test_geometry_changes_state_separation():
    r256=evaluate(Config(p=Params(dim=256)),seeds=20,run_conditions=('adaptive',))
    r1024=evaluate(Config(p=Params(dim=1024)),seeds=20,run_conditions=('adaptive',))
    sep=lambda r:r['conditions']['adaptive']['readouts']['C_mem']['separation']
    assert sep(r256)>sep(r1024)+.2


def test_bounded_local_gain_and_reproducible_seed():
    cfg=Config()
    chain=ContinuousChain(cfg)
    for _ in range(1200):
        chain.step(pulse_a=10.)
    assert all(cfg.min_gain<=c.gain<=cfg.max_gain for c in chain.cells[1:])
    x=trial(+1,123,cfg)
    y=trial(+1,123,cfg)
    assert x==y
