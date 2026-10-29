"""Direct-field leakage audit for the *rejected* external-source version.

Important: an external field source near A can directly push B/C even if
A's own emission is off. A source-driven B/C memory response is therefore
NOT evidence of an A-emitted relay. See RESULTS.md.
"""
import argparse
import math
import numpy as np
from .assay import CENTERS,wrap
from .physical_subset import Field

def trial(seed=0,amp=.3,disable_a=False):
    initial=Field(pair_centers=CENTERS,seed=seed)
    for _ in range(3000):initial.step()
    driven=initial.fork();sham=initial.fork()
    if disable_a:
        for w in (driven,sham):
            w.emit_mask[[0,w.npairs]]=0.
    point=(CENTERS[0][0],CENTERS[0][1]+2.5)
    for _ in range(120):
        driven.step((point,amp*math.cos(.11*driven.t)))
        sham.step()
    for _ in range(300):
        driven.step();sham.step()
    return wrap(driven.phases()-sham.phases())[1:].tolist()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seeds',type=int,default=10)
    a=p.parse_args()
    for seed in range(a.seeds):
        on=np.array(trial(seed))
        off=np.array(trial(seed,disable_a=True))
        print(seed,'A-emitting',on,'A-emission-disabled',off,
              'off/on norm',float(np.linalg.norm(off)/(np.linalg.norm(on)+1e-12)))

if __name__=='__main__':main()
