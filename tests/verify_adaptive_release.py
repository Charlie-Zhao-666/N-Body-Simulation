"""Behavioral publication checks, not a performance benchmark.

From the repository root: python tests/verify_adaptive_release.py --gui --output checks.json
Optional --gpu exercises CUDA; --production compares an explicitly supplied local source.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import queue
import sys
import threading
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', action='store_true')
parser.add_argument('--gui', action='store_true')
parser.add_argument('--production', type=Path)
parser.add_argument('--output', type=Path)
args = parser.parse_args()

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

m = load(ROOT/'interactive_nbody_lab.py', 'release_app')
production = load(args.production, 'production_app') if args.production else None
variants = ('work_step', 'work_step_align', 'work_step_average', 'work_step_predicted')
assert set(m.ADAPTIVE_ALGORITHMS) == {'timescale', 'acceleration_ratio', *variants}
orbit = m.STABLE_ORBITS[4]
base = dict(initial_mode='orbit', orbit_index=4, seed=5, particle_count=3,
            dt=orbit['dt'], total_time=1.3, method='1', sample_every=10,
            compute_backend='cpu_array', cpu_threads=1, adaptive_enabled=True,
            adaptive_algorithm='work_step', work_step_factor=10., work_max_power=3,
            work_threshold=.01, work_preprobe_enabled=True)
checks = {}
for retired in ('energy_ratio', 'work_skill_test', 'work_skill_test_align', 'work_skill_test_wide'):
    for validate in (m.validate_config, lambda config: m.prepare_batch_plan(dict(base=config, runs=[{}]))):
        try:
            validate(dict(base, adaptive_algorithm=retired))
        except ValueError:
            pass
        else:
            raise AssertionError(f'Retired mode accepted: {retired}')
checks['retired_modes_rejected_in_main_and_batch'] = True

a0, a1, prior = np.array([[2.,0.,0.]]), np.array([[100.,0.,0.]]), np.array([[1.,0.,0.]])
np.testing.assert_array_equal(m.select_work_acceleration(a0,a1),a1)
np.testing.assert_array_equal(m.select_work_acceleration(a0,a1,aligned=True),a0)
np.testing.assert_array_equal(m.select_work_acceleration(a0,-a0,average=True),np.zeros_like(a0))
np.testing.assert_array_equal(m.select_work_acceleration(a0,a1,predicted=True,previous_acceleration=prior),[[2.5,0,0]])
np.testing.assert_array_equal(m.select_work_acceleration(a0,-a1,predicted=True,previous_acceleration=prior),[[2.5,0,0]])
np.testing.assert_array_equal(m.select_work_acceleration(a0,a1,predicted=True),a0)
c=m.WorkStepController(1.,masses=[1.],energy_scale=10.,predicted=True)
np.testing.assert_allclose(c.indicators(a1,np.array([[3.,0,0]]),.1,
    initial_acceleration=a0,final_velocity=None,previous_acceleration=prior),[.078125],rtol=0,atol=1e-16)
checks['acceleration_vectors_cancellation_predicted_2_5_and_first_step'] = True

class ProbeSpy(m.WorkStepController):
    def __init__(self):
        super().__init__(1.,masses=[1.],energy_scale=1.,predicted=True)
        self.used=[]
    def indicators(self,*pos,**kw):
        self.used.append(m.select_work_acceleration(kw['initial_acceleration'],pos[0],
            predicted=True,previous_acceleration=kw['previous_acceleration']).copy())
        return super().indicators(*pos,**kw)
spy=ProbeSpy()
values=iter([3.,4.,1.])
event=m.probe_work_encounter(0,1,.1,a0,
    lambda h:(np.array([[next(values),0,0]]),0.),lambda:np.zeros((1,3)),spy,prior_acceleration=prior)
np.testing.assert_allclose(np.array(spy.used)[:,0,0],[2.5,3.5,4.5])
assert event['status']=='peak_and_return'
checks['disposable_probe_has_separate_history'] = True

from nbody_array_cpu import warmup_cpu_backend
warmup_cpu_backend(False)
runs=[]
for method in ('1','2'):
    for algorithm in variants:
        cfg=dict(base,method=method,adaptive_algorithm=algorithm)
        result=m.run_simulation(cfg)
        meta=result['metadata']
        assert meta['work_probe_trial_evaluations']>0
        assert meta['discarded_trial_steps']>0
        assert meta['work_trial_evaluations']==meta['steps']+meta['discarded_trial_steps']+meta['work_probe_trial_evaluations']
        assert result['record_times'][-1]==cfg['total_time']
        assert np.all(np.diff(result['record_times'])>0)
        a,v,h=result['accelerations'],result['velocities'],result['step_sizes']
        mass=np.array([s['mass'] for s in meta['initial_stars']])
        expected=[]
        for index in range(len(h)-1):
            A=(a[index+1] if algorithm=='work_step' else
               a[index] if algorithm=='work_step_align' else
               (a[index]+a[index+1])/2 if algorithm=='work_step_average' else
               a[index]+.5*(a[index]-a[max(0,index-1)]))
            power=np.abs(np.sum(A*v[index],axis=-1))
            expected.append(mass*(power*h[index+1]+.5*np.sum(A*A,axis=-1)*h[index+1]**2)/meta['energy_scale'])
        exported=m.projection_export_arrays(a,v,result['record_times'],h,meta)
        np.testing.assert_allclose(exported['work_step_indicator'][1:],expected,rtol=6e-15,atol=1e-30)
        if production:
            before=production.run_simulation(cfg)
            for key in ('record_times','step_sizes','energies','trajectories','velocities','accelerations'):
                np.testing.assert_array_equal(before[key],result[key],err_msg=f'{algorithm}/{method}/{key}')
        runs.append(dict(method=method,algorithm=algorithm,steps=meta['steps'],
                         rejected=meta['discarded_trial_steps'],probe_trials=meta['work_probe_trial_evaluations']))
        print(json.dumps(runs[-1]),flush=True)
checks['four_variants_both_integrators_probe_rollback_export_and_cost']=runs
checks['production_bitwise_parity']=production is not None

backends=['cpu','cpu_parallel']+(['gpu'] if args.gpu else [])
for backend in backends:
    for method in ('1','2'):
        r=m.run_simulation(dict(base,total_time=.03,method=method,compute_backend=backend,
                                cpu_threads=2,adaptive_algorithm='work_step_predicted'))
        assert np.isfinite(r['energies']).all()
checks['additional_backend_smoke'] = backends

caps=[]
for method in ('1','2'):
    r=m.run_simulation(dict(base,total_time=.02,dt=.001,method=method,work_preprobe_enabled=False,
                           work_step_factor=2.,work_max_power=1,work_threshold=1e-12))
    meta=r['metadata']
    assert meta['work_exceeded_steps']>0 and meta['discarded_trial_steps']>0
    assert min(r['step_sizes'][1:-1])>=.0005
    caps.append(dict(method=method,flagged=meta['work_exceeded_steps']))
checks['cap_accepts_and_flags_exceptions']=caps

for plan in ('work-step-demo.json','work-scale-four-methods.json'):
    configs=m.prepare_batch_plan(json.loads((ROOT/'plans'/plan).read_text(encoding='utf-8')))
    assert len(configs)==(8 if 'four-methods' in plan else 2)
checks['published_plans_load']=True

if args.gui:
    import tkinter as tk
    done=threading.Event();done.set()
    m._CPU_STARTUP=SimpleNamespace(messages=queue.Queue(),done=done,stop=threading.Event(),errors={},worker=None)
    root=tk.Tk();root.withdraw()
    app=m.NBodyApp(root);app.history_migration_pending=False
    app.adaptive_var.set(True)
    for algorithm in variants:
        app.adaptive_algorithm_var.set(m.ADAPTIVE_ALGORITHMS[algorithm]);app.sync_fields()
        assert app.read_config()['adaptive_algorithm']==algorithm
    editor=m.BatchPlanEditor(app,m.validate_config(dict(base,adaptive_algorithm='work_step_predicted')))
    editor.window.withdraw()
    assert all(m.ADAPTIVE_ALGORITHMS[x] in editor.widgets['adaptive_algorithm']['values'] for x in variants)
    editor.window.destroy()
    viewroot=tk.Toplevel(root);viewroot.withdraw()
    viewer=m.NBodyViewer(result['trajectories'],result['accelerations'],result['energies'],result['time_step'],
        result['diagnostic_times'],result['diagnostics'],result['metadata'],root=viewroot,
        record_times=result['record_times'],step_sizes=result['step_sizes'],velocities=result['velocities'])
    if not hasattr(viewer,'derivative_figure'):viewer.build_acceleration_derivative()
    viewer.refresh_acceleration_derivative()
    assert viewer.work_predicted and 'a_pred' in viewer.work_indicator_formula
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        viewer.derivative_figure.savefig(args.output.parent/'publication-predicted-monitor.png',dpi=100)
    viewroot.destroy();app.closed=True;root.destroy()
checks['real_tk_main_batch_and_viewer']=args.gui

report=dict(date='2026-11-02',status='passed',purpose='Release behavior checks, not an accuracy or speed ranking',
            python=sys.version.split()[0],numpy=np.__version__,checks=checks)
if args.output:
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,indent=2))
