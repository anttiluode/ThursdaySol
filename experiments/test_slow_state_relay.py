"""Run from repo root: python -m pytest -q experiments/test_slow_state_relay.py"""
from experiments.slow_state_relay import Chain, Params, evaluate, prepare, trial


def test_three_real_independent_links_and_staged_gating():
    chain, _ = prepare(+1, 7, Params())
    assert not chain.ab and not chain.bc
    assert chain.cells[0].memory == 0  # sender silenced at checkpoint
    assert chain.cells[1].output == .6  # B fast output matched
    assert chain.cells[1].memory > 0   # history survives
    assert abs(chain.cells[2].memory) < .05  # C was gated during writing


def test_local_memory_state_is_bounded_under_strong_drive():
    chain = Chain()
    for _ in range(1000):
        chain.step(pulse_a=100.)
    assert all(-1 <= c.memory <= 1 for c in chain.cells)


def test_relay_requires_intermediate_memory_and_live_b_c_connection():
    p = Params()
    plus = trial(+1, 101, p)
    minus = trial(-1, 102, p)
    assert plus['c_memory'] > .1 and minus['c_memory'] < -.1
    for ablation in ({'erase_b': True}, {'cut_bc': True}):
        plus0 = trial(+1, 101, p, **ablation)
        minus0 = trial(-1, 102, p, **ablation)
        assert abs(plus0['c_memory']-minus0['c_memory']) < .02


def test_c_state_changes_identical_later_probe_readout():
    p = Params()
    r = evaluate(p, seeds=40)['conditions']
    assert r['normal']['heldout_accuracy_C_probe_output'] >= .95
    assert abs(r['normal']['probe_interaction']) > .005
    assert abs(r['erase_C_memory']['probe_interaction']) < .001
    assert .3 <= r['erase_C_memory']['heldout_accuracy_C_probe_output'] <= .7


def test_robustness_is_bounded_and_not_unconditional():
    assert evaluate(Params(noise_sd=.03), seeds=40)['conditions']['normal']['heldout_accuracy_C_probe_output'] >= .75
    assert evaluate(Params(noise_sd=.1), seeds=40)['conditions']['normal']['heldout_accuracy_C_probe_output'] < .75


def test_observer_geometry_modulates_transfer_only_with_added_sensor_gate():
    a = evaluate(Params(dim=256),seeds=20)['conditions']['normal']
    b = evaluate(Params(dim=1024),seeds=20)['conditions']['normal']
    sep=lambda x: x['mean_C_memory_plus']-x['mean_C_memory_minus']
    assert sep(a) > sep(b)+.1
    c = evaluate(Params(dim=256, sensor_mod=0),seeds=20)['conditions']['normal']
    d = evaluate(Params(dim=1024, sensor_mod=0),seeds=20)['conditions']['normal']
    assert abs(sep(c)-sep(d)) < 1e-12
