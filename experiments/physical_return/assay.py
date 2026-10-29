"""Phase-only source -> reciprocal wave field -> hidden B/C D -> A assay.

Run: python -m experiments.physical_return.assay --seeds 12
No external field forcing during writes; sole write is A phase intervention.
An identical later external probe interrogates A. This is NOT a directed
three-hop relay; the KG field propagates reciprocally in two dimensions.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from .physical_subset import Field

CENTERS=((24.,21.),(28.,26.),(20.,26.))

def wrap(x):
    return np.angle(np.exp(1j*x))

def transplant(to, source, k, mode='D'):
    """Move only D or only S in one pair, preserving its complementary mode."""
    a=k; b=to.npairs+k
    coeff=1 if mode=='S' else -1
    target=wrap(source.theta[a]+coeff*source.theta[b])
    before=wrap(to.theta[a]+coeff*to.theta[b])
    diff=wrap(target-before)
    to.theta[a]=(to.theta[a]+diff/2)%(2*math.pi)
    to.theta[b]=(to.theta[b]+coeff*diff/2)%(2*math.pi)

def prepare(seed=0,kick=.6,hold=800,emit_A=True):
    w=Field(pair_centers=CENTERS,seed=seed)
    for _ in range(3000):w.step()
    control=w.fork();written=w.fork()
    written.theta[0]=(written.theta[0]+kick)%(2*math.pi)
    if not emit_A:
        for v in (written,control):
            v.emit_mask[[0,v.npairs]]=0.
    for _ in range(hold):
        written.step();control.step()
    return written,control

def variant(written,control,which):
    v=control.fork()
    if which in ('D_B','D_BC','DS_BC','D_BC_emit_cut'):
        transplant(v,written,1,'D')
    if which in ('D_C','D_BC','DS_BC','D_BC_emit_cut'):
        transplant(v,written,2,'D')
    if which in ('S_BC','DS_BC'):
        for k in (1,2):transplant(v,written,k,'S')
    if which in ('control_emit_cut','D_BC_emit_cut'):
        v.emit_mask[[1,2,v.npairs+1,v.npairs+2]]=0.
    return v

def response(w,amp=.15,duration=950):
    """Same subsequent probe minus no-probe, on unwrapped A sum phase."""
    a=w.fork();b=w.fork();origin=float(w.sums()[0]);aa=[];bb=[]
    position=(CENTERS[0][0],CENTERS[0][1]+2.5)
    for t in range(duration):
        forcing=(position,amp*math.cos(.11*a.t)) if 100<=t<180 else None
        a.step(forcing);b.step()
        aa.append(float(a.sums()[0]));bb.append(float(b.sums()[0]))
    z1=np.unwrap([origin]+aa)[-1]
    z0=np.unwrap([origin]+bb)[-1]
    return float(z1-z0)

def trial(seed=0,kick=.6,hold=800):
    written,control=prepare(seed,kick,hold)
    names=('control','D_B','D_C','D_BC','S_BC','DS_BC',
           'control_emit_cut','D_BC_emit_cut')
    variants={k:variant(written,control,k) for k in names}
    baseline=response(variants['control'])
    values={k:response(v) for k,v in variants.items()}
    effects={k:values[k]-baseline for k in ('D_B','D_C','D_BC','S_BC','DS_BC')}
    effects['no_BC_emission']=values['D_BC_emit_cut']-values['control_emit_cut']
    v=variants['D_BC']
    return {'seed':seed,
            'dD_BC':wrap(written.phases()-control.phases())[1:].tolist(),
            'dS_BC':wrap(written.sums()-control.sums())[1:].tolist(),
            'effects':effects,
            'field_matched':bool(np.array_equal(control.psi,v.psi) and
                                 np.array_equal(control.prev,v.prev)),
            'A_matched':bool(np.array_equal(control.theta[[0,control.npairs]],v.theta[[0,control.npairs]])),
            'sum_BC_matched':bool(np.max(np.abs(wrap(control.sums()[1:]-v.sums()[1:])))<1e-12)}

def evaluate(seeds=12):
    rows=[trial(seed) for seed in range(seeds)]
    stats={}
    for key in rows[0]['effects']:
        samples=np.array([r['effects'][key] for r in rows])
        sem=float(samples.std(ddof=1)/math.sqrt(seeds))
        stats[key]={'mean':float(samples.mean()),'mean_abs':float(np.abs(samples).mean()),
                    'max_abs':float(np.abs(samples).max()),
                    'normal_approx_95ci':[float(samples.mean()-1.96*sem),
                                          float(samples.mean()+1.96*sem)],
                    'positive':int(np.sum(samples>1e-10)),
                    'negative':int(np.sum(samples< -1e-10))}
    return {'model':'pinned NumPy port of oscillator+KG laws; no extra memory register',
            'topology':'reciprocal triangle; NOT demonstrated directed A->B->C->A relay',
            'protocol':{'kick_A':.6,'hold':800,'warm':3000,'N':48,'delta':.015,
                        'probe_amp':.15,'probe_freq':.11,'probe_start':100,
                        'probe_duration':80,'readout_ticks':950,'phase_noise':0.},
            'stats':stats,'trials':rows}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seeds',type=int,default=12)
    p.add_argument('--out',default='experiments/physical_return/results/native_return.json')
    a=p.parse_args()
    result=evaluate(a.seeds)
    path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n')
    for k,v in result['stats'].items():
        print(f'{k:18} mean={v["mean"]:+.7f} mean|effect|={v["mean_abs"]:.7f} '
              f'ci95={v["normal_approx_95ci"]} +/-={v["positive"]}/{v["negative"]}')

if __name__=='__main__':main()
