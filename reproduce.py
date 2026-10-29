"""Re-run the fixed ECG three-loop protocol and declared controls."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from three_loop import assay

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
RESULTS.mkdir(parents=True, exist_ok=True)
DEFAULT = dict(initial=.6, pulse_amp=.5, warm=140, duration=12, hold=8,
               probe=.15, steps=80, dim=256, gain=.2)

if __name__ == '__main__':
    main = assay(**DEFAULT)
    (RESULTS/'baseline.json').write_text(json.dumps(main,indent=2)+'\n')
    tests = {
        'no_A_write': assay(**{**DEFAULT,'pulse_amp':0.}),
        'long_hold_30': assay(**{**DEFAULT,'hold':30}),
        'long_hold_60': assay(**{**DEFAULT,'hold':60}),
        'observer_1024': assay(**{**DEFAULT,'dim':1024}),
        'observer_2048': assay(**{**DEFAULT,'dim':2048}),
    }
    noise_trials = [assay(**DEFAULT,noise=.003,seed=s) for s in range(1,9)]
    tests['paired_noise_0.003_seeds_1_to_8'] = noise_trials
    summarize=lambda r: {k:v for k,v in r.items() if k!='traces'}
    summaries={k:([summarize(v) for v in rows] if isinstance(rows,list) else summarize(rows))
               for k,rows in tests.items()}
    summaries['baseline'] = summarize(main)
    (RESULTS/'controls.json').write_text(json.dumps(summaries,indent=2)+'\n')
    for key in ['no_A_write','long_hold_30','long_hold_60','observer_1024','observer_2048']:
        r=tests[key]
        print(key,'C_probe_interaction_peak =',round(r['memory_graft']['C_probe_interaction_peak'],10))
    xs=[r['memory_graft']['C_probe_interaction_peak'] for r in noise_trials]
    print('Noise 0.003, 8 paired preparations: min/median/max C interaction',
          min(xs),float(np.median(xs)),max(xs))

    # One plot per figure, no specified colors/styles.
    fig,ax=plt.subplots(figsize=(10,4.7))
    ax.plot(main['traces']['graft_C_reference'],label='C with sham B history')
    ax.plot(main['traces']['graft_C_history'],label='C with A-written B history')
    ax.set(xlabel='Ticks after matching visible state and disconnecting A → B',
           ylabel='Loop C controller output',
           title='Identical present output and same later probe; different B history')
    ax.legend()
    fig.tight_layout();fig.savefig(RESULTS/'matched_history_C.png',dpi=150);plt.close(fig)

    fig,ax=plt.subplots(figsize=(10,4.7))
    ax.plot(main['traces']['C_probe_interaction'],label='Deterministic, no input noise')
    for s,r in enumerate(noise_trials,1):
        ax.plot(r['traces']['C_probe_interaction'],alpha=.35,label='Paired-noise seeds' if s==1 else None)
    ax.set(xlabel='Ticks after matched-state fork',ylabel='History × probe interaction at C',
           title='A large deterministic effect does not survive small input noise')
    ax.legend();fig.tight_layout();fig.savefig(RESULTS/'robustness.png',dpi=150);plt.close(fig)
