"""Run: python -m pytest -q test_three_loop.py"""
import math
import numpy as np
import cv2
from three_loop import ECGNode, assay, spatial_samples


def test_observer_geometry_changes_which_pixels_are_read():
    assert spatial_samples(48, 256) == (0., 0., 0., 1.)
    assert spatial_samples(48, 1024) == (0., 0., 0., 0.)


def test_spatial_samples_exactly_match_full_original_math():
    y, x = np.mgrid[0:256, 0:256]
    for size in (7, 48, 51, 90, -14):
        image = np.stack([((x//size+y//size)%2)]*3, axis=-1).astype(np.float32)
        gray = image.mean(axis=2)
        for dim in (256, 1024, 2048):
            side = math.ceil(math.sqrt(dim))
            v = cv2.resize(gray, (side,side), interpolation=cv2.INTER_AREA).flatten()[:dim]
            if np.max(np.abs(v))>0:
                v = v/np.max(np.abs(v))
            assert np.allclose(spatial_samples(size, dim), v[:4])


def test_controller_equation_first_step():
    n = ECGNode()
    n.step(1.0)
    assert math.isclose(n.output, .6+math.tanh(1.6))
    assert n.regime == 1.
    assert len(n.history) == 1


def test_zero_pulse_cannot_create_history_difference():
    r = assay(.6,0,140,12,8,.15,80)
    assert r['at_checkpoint']['B_history_l2'] == 0
    assert r['natural']['C_peak_delta'] == 0
    assert r['memory_graft']['C_peak_delta'] == 0


def test_b_history_alone_can_causally_change_c_in_one_deterministic_preparation():
    r = assay(.6,.5,140,12,8,.15,80)
    assert r['at_checkpoint']['B_history_l2'] > 0
    assert r['memory_graft']['B_first_delta'] == 0
    assert r['memory_graft']['C_first_delta'] == 0
    assert r['memory_graft']['C_probe_interaction_peak'] > .5
    assert r['memory_graft']['C_no_BC_peak_delta'] == 0
    assert r['memory_graft']['C_no_probe_peak_delta'] < .001


def test_history_is_not_necessarily_readable_after_long_hold():
    r = assay(.6,.5,140,12,60,.15,80)
    assert r['at_checkpoint']['B_history_l2'] > .01
    assert r['memory_graft']['C_probe_interaction_peak'] < 1e-6


def test_dimension_change_blocks_this_particular_readout():
    r = assay(.6,.5,140,12,8,.15,80,dim=1024)
    assert r['at_checkpoint']['B_history_l2'] > 0
    assert r['memory_graft']['C_probe_interaction_peak'] < 1e-6


def test_small_shared_noise_breaks_large_deterministic_response():
    vals=[]
    for seed in range(1, 9):
        r=assay(.6,.5,140,12,8,.15,80,noise=.003,seed=seed)
        vals.append(r['memory_graft']['C_probe_interaction_peak'])
        assert r['memory_graft']['C_no_BC_peak_delta'] == 0
    assert max(vals) < .001
