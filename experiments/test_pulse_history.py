"""Run from ThursdaySol root: python -m pytest -q experiments/test_pulse_history.py"""
from dataclasses import replace
import math
import pytest

from experiments.pulse_history import (Protocol, sample_stream,
                                      evaluate_stream, evaluate_paired_context)
from experiments.slow_state_relay import REST


def test_symbols_must_be_signed():
    with pytest.raises(ValueError):
        sample_stream((0, 1), 4)


def test_repeatable_with_fixed_seed_and_bounded_two_pulse_state():
    p=Protocol(noise_sd=.003)
    x=sample_stream((1,-1),seed=43,protocol=p)
    y=sample_stream((1,-1),seed=43,protocol=p)
    assert x == y
    assert -1 <= x['C_memory_pre_erase'] <= 1
    assert abs(x['C_memory_pre_erase']) < .4  # unlike prior saturated +/- one-bit memory


def test_paired_history_changes_increment_not_only_baseline():
    result=evaluate_paired_context(n_pairs=40)
    d=result['differences']['increment']
    assert d['paired_difference_mean'] < -.003
    assert d['normal_approx_ci95'][1] < 0
    assert d['fraction_negative'] >= .9
    # This is explicitly a difference-in-differences at C, with matched
    # *current* output and B's current input absent during diagnostic readout.


@pytest.mark.parametrize('control', ['erase_C','erase_B','cut_BC','latest_only','no_probe'])
def test_causal_controls_null_paired_probe_interaction(control):
    result=evaluate_paired_context(n_pairs=12,control=control)
    assert abs(result['differences']['increment']['paired_difference_mean']) < 1e-12


def test_fixed_gain_also_produces_history_effect():
    r=evaluate_paired_context(n_pairs=40,adaptive=False)
    assert r['differences']['increment']['paired_difference_mean'] < -.002


def test_heldout_stream_decodes_early_bit_but_limited_recent_bit():
    r=evaluate_stream(length=2,per_word=40)
    bits=r['bit_decoding']['C_probe_increment']
    assert bits[0] > .7
    assert bits[1] < .7
    assert r['word_decoding']['C_probe_increment'] > .25


def test_no_history_stream_control_earlier_bit_at_chance():
    r=evaluate_stream(length=2,per_word=40,control='latest_only')
    assert .3 <= r['bit_decoding']['C_probe_increment'][0] <= .7


def test_high_noise_breaks_independently_decodable_stream():
    r=evaluate_stream(length=2,per_word=40,protocol=Protocol(noise_sd=.03))
    assert r['bit_decoding']['C_probe_increment'][0] < .65


def test_three_pulse_capacity_is_limited():
    r=evaluate_stream(length=3,per_word=24)
    assert r['bit_decoding']['C_probe_increment'][0] > .65
    assert .3 < r['bit_decoding']['C_probe_increment'][2] < .7
    assert r['word_decoding']['C_probe_increment'] < .35  # chance is 1/8


def test_no_probe_has_exactly_zero_increment():
    x=sample_stream((-1,1),33,ignore_probe=True)
    assert x['C_probe_increment'] == 0.
