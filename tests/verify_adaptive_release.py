"""Short behavioral checks, not a performance benchmark. Run from the repo root.

python tests/verify_adaptive_release.py --gui --gpu --output checks.json
Optional --baseline points to a pre-change application for exact regression checks.
"""
from pathlib import Path
import argparse
import ast
import importlib.util
import json
import sys
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser()
parser.add_argument('--baseline', type=Path)
parser.add_argument('--gpu', action='store_true')
parser.add_argument('--gui', action='store_true')
parser.add_argument('--output', type=Path)
args = parser.parse_args()

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

m = load(ROOT/'interactive_nbody_lab.py', 'release_app')
old = load(args.baseline, 'baseline_app') if args.baseline else None
assert set(m.ADAPTIVE_ALGORITHMS) == {'timescale', 'acceleration_ratio', 'work_step'}
butterfly = next(i for i, item in enumerate(m.STABLE_ORBITS) if item['name'].startswith('Butterfly I ('))
base = dict(initial_mode='orbit', orbit_index=butterfly, seed=1, particle_count=3,
            dt=.001, total_time=.02, method='1', sample_every=3,
            compute_backend='cpu', cpu_threads=2, adaptive_enabled=True,
            adaptive_algorithm='work_step', work_step_factor=2., work_max_power=1,
            work_threshold=1e-12)
try:
    m.validate_config(dict(base, adaptive_algorithm='energy_ratio'))
    raise AssertionError('Removed method must not run')
except ValueError as exc:
    assert 'removed' in str(exc)
try:
    m.prepare_batch_plan(dict(base=base, runs=[dict(adaptive_algorithm='energy_ratio')]))
    raise AssertionError('Removed method must not load as a runnable batch')
except ValueError:
    pass
legacy = m.prepare_batch_plan(dict(base=dict(base, energy_relative_tolerance=.1, energy_max_rejections=7), runs=[{}]))[0]['config']
assert legacy['adaptive_algorithm'] == 'work_step'
assert 'energy_relative_tolerance' not in legacy and 'energy_max_rejections' not in legacy

c = m.WorkStepController(1., 2., 3, 1.)
c.level = 2
assert c.propose_recovery(2.)[0] == .25
assert c.propose_recovery(1.99)[0] == .5
assert c.propose_recovery(0.)[0] == 1.
assert c.propose_recovery(0.)[0] == 1.

backends = ['cpu', 'cpu_array', 'cpu_parallel'] + (['gpu'] if args.gpu else [])
checks = []
for method in ('1', '2'):
    for backend in backends:
        config = dict(base, method=method, compute_backend=backend)
        result = m.run_simulation(config)
        meta = result['metadata']
        assert meta['discarded_trial_steps'] > 0
        assert meta['work_exceeded_steps'] > 0
        assert meta['work_exceeded_intervals']
        assert min(result['step_sizes'][1:-1]) >= .0005
        assert result['record_times'][-1] == .02
        assert not any(key.startswith('energy_rejection') for key in meta)
        if old:
            before = old.run_simulation(config)
            for key in ('record_times', 'step_sizes', 'trajectories', 'velocities', 'accelerations', 'energies'):
                np.testing.assert_array_equal(before[key], result[key])
            assert before['metadata']['work_exceeded_intervals'] == meta['work_exceeded_intervals']
        # Independent replay of accepted steps catches trial-state leakage.
        stars = m.make_initial_state(config)
        integrate = m.leapfrog if method == '1' else m.rk4
        for index, h in enumerate(result['step_sizes'][1:], 1):
            integrate(stars, h)
            np.testing.assert_allclose([s['pos'] for s in stars], result['trajectories'][:, index], rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose([s['v'] for s in stars], result['velocities'][index], rtol=1e-10, atol=1e-12)
        checks.append(dict(method=method, backend=backend, steps=meta['steps'],
                           rejected=meta['discarded_trial_steps'], flagged=meta['work_exceeded_steps']))
if old:
    for method in ('1', '2'):
        for mode in ('fixed', 'timescale', 'acceleration_ratio'):
            config = dict(base, method=method, adaptive_enabled=mode!='fixed',
                          adaptive_algorithm='timescale' if mode=='fixed' else mode)
            before, after = old.run_simulation(config), m.run_simulation(config)
            for key in ('record_times', 'step_sizes', 'trajectories', 'velocities', 'accelerations', 'energies'):
                np.testing.assert_array_equal(before[key], after[key])
    def definitions(path):
        return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(path.read_text(encoding='utf-8-sig')).body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
    a, b = definitions(args.baseline), definitions(ROOT/'interactive_nbody_lab.py')
    for name in ('WorkStepController', 'AccelerationRatioController', 'leapfrog', 'rk4', 'GPUState', 'net_accelerations', 'adaptive_step_size'):
        assert a[name] == b[name], name

if args.gui:
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    try:
        with patch.object(m.NBodyApp, 'save_pending_history'):
            window = m.NBodyApp(root=root)
        window.adaptive_var.set(True)
        for code in m.ADAPTIVE_ALGORITHMS:
            window.adaptive_algorithm_var.set(m.ADAPTIVE_ALGORITHMS[code])
            window.sync_fields()
            assert window.read_config()['adaptive_algorithm'] == code
        editor = m.BatchPlanEditor(window, window.read_config())
        editor.window.withdraw()
        assert editor.read_current()['config']['adaptive_algorithm'] == 'work_step'
        assert not any(key.startswith('energy_') for key in editor.vars)
        editor.window.destroy()
        viewer_root = tk.Toplevel(root)
        viewer_root.withdraw()
        viewer = m.NBodyViewer(root=viewer_root, **result)
        assert viewer.metadata['work_exceeded_intervals']
        assert viewer.drift_axes.get_yscale() == 'linear'
        viewer.diagnostics_png()
        viewer.close()
        window.cpu_warmup_worker.join(timeout=15)
    finally:
        window.close()

report = dict(date='2026-10-20', purpose='Behavioral release regression, not a speed/accuracy ranking',
              gpu_tested=args.gpu, gui_tested=args.gui, baseline_exact_regression=old is not None,
              removed_energy_ratio_rejected=True, controller_recovery_boundaries=True,
              rollback_replayed=True, runs=checks)
if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
