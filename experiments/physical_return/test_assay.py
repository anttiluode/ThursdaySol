"""Run: python -m pytest -q experiments/physical_return/test_assay.py"""
import numpy as np
from .assay import prepare, trial, variant, wrap

def test_kick_reaches_neighbors_only_through_A_emission():
    written,control=prepare(seed=1,kick=.6,hold=800)
    assert np.max(np.abs(wrap(written.phases()-control.phases())[1:]))>.03
    silent,sham=prepare(seed=1,kick=.6,hold=800,emit_A=False)
    assert np.array_equal(silent.phases()[1:],sham.phases()[1:])

def test_counterfactual_matches_A_and_field_and_BC_sum_modes():
    written,control=prepare(seed=1,kick=.6,hold=800)
    v=variant(written,control,'D_BC')
    assert np.array_equal(control.psi,v.psi)
    assert np.array_equal(control.prev,v.prev)
    assert np.array_equal(control.theta[[0,control.npairs]],
                          v.theta[[0,control.npairs]])
    assert max(np.abs(wrap(control.sums()[1:]-v.sums()[1:])))<1e-12
    assert max(np.abs(wrap(control.phases()[1:]-v.phases()[1:])))>.03

def test_difference_of_probe_response_requires_BC_emission():
    r=trial(seed=1,kick=.6,hold=800)
    assert r['field_matched'] and r['A_matched'] and r['sum_BC_matched']
    assert r['effects']['D_BC']>.002
    assert r['effects']['no_BC_emission']==0.

def test_zero_phase_kick_changes_nothing():
    written,control=prepare(seed=1,kick=0.,hold=800)
    assert np.array_equal(written.theta,control.theta)
    assert np.array_equal(written.psi,control.psi)
