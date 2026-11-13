"""Automatic analytic two-body comparison page; uses the viewer's shared timeline."""
import numpy as np
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from nbody_kepler import binary_state
from nbody_two_body import TwoBodyReference


def reference_positions(times, metadata):
    error = metadata.get('position_error') or {}
    if error.get('reference_model') == 'universal_kepler_initial_state':
        return TwoBodyReference(metadata['initial_stars'], metadata.get('G', 1.)).positions(times)
    return binary_state(times, error['eccentricity'])[0]


def position_export_arrays(trajectories, times, metadata):
    error = metadata.get('position_error') or {}
    if error.get('reference_kind') != 'analytic_kepler':
        return {}
    analytic = reference_positions(times, metadata)
    delta = np.asarray(trajectories).transpose(1, 0, 2)-analytic
    length_scale = error.get('length_scale', (1+error['eccentricity'])/2)
    return dict(analytic_positions=analytic, position_error_vectors=delta,
                position_error_norms=np.linalg.norm(delta, axis=2),
                position_error_normalized=np.linalg.norm(delta, axis=2)/length_scale)


class PositionAnalysisMixin:
    def build_position_analysis(self):
        self.position_tab = None
        error = self.metadata.get('position_error') or {}
        if error.get('reference_kind') != 'analytic_kepler':
            return
        self.position_analysis_arrays = position_export_arrays(self.trajectories, self.times, self.metadata)
        self.position_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.position_tab, text='Position error · Analytic two-body')
        controls = ttk.Frame(self.position_tab, padding=(6, 4))
        controls.pack(fill='x')
        ttk.Label(controls, text='Error Y axis:').pack(side='left')
        self.position_error_scale = tk.StringVar(value='linear')
        for label, value in (('Linear', 'linear'), ('Log (symlog)', 'symlog')):
            ttk.Radiobutton(controls, text=label, value=value, variable=self.position_error_scale,
                            command=self.refresh_position_analysis).pack(side='left', padx=5)
        ttk.Button(controls, text='Save plot PNG', command=self.save_image).pack(side='right')
        self.position_figure = Figure(figsize=(12, 8), dpi=100, constrained_layout=True)
        self.position_axes = self.position_figure.subplots(3, 1, sharex=True)
        self.position_figure.suptitle(
            f"Analytic two-body reference | {self.metadata['method_name']}\n"
            f"Position RMSE = {error['rmse']:.5e}   NRMSE = {error['nrmse']:.5e}   "
            + (f"(time-weighted; all {error['samples']:,} records)" if error.get('metric_version') == 2
               else f"({error['samples']:,} uniform-time samples; legacy Hermite metric)"), fontsize=10)
        self.position_coordinate_lines = []
        for component, color in zip('xyz', ('#2563eb', '#dc2626', '#059669')):
            numerical, = self.position_axes[0].plot([], [], color=color, lw=1.2, label=f'{component} numerical')
            exact, = self.position_axes[0].plot([], [], color=color, lw=1, ls='--', label=f'{component} analytic')
            self.position_coordinate_lines.append((numerical, exact))
        self.position_axes[0].set_ylabel('Position (code length)')
        for axis, key, label in zip(self.position_axes[1:],
                ('position_error_norms','position_error_normalized'),
                ('3D position error (code length)','3D error / initial RMS radius')):
            values = self.position_analysis_arrays[key]
            series = {}
            for i in range(self.n):
                line, = axis.plot([], [], lw=1, label=f'Star {i+1}')
                self.set_time_series(line, self.times, values[:, i])
                series[f'Star {i+1}'] = values[:, i]
            axis.axhline(0, color='#64748b', ls='--', lw=.8, label='Exact agreement (zero error)')
            axis.set_ylabel(label)
            axis.set_title('Pointwise error at every accepted record' if key == 'position_error_norms'
                           else 'Pointwise normalized error (dimensionless)', fontsize=9)
            self.register_hover(axis, self.times, series, title=label)
        self.position_axes[-1].set_xlabel('Time (code units)')
        self.position_cursors = []
        for axis in self.position_axes:
            axis.grid(alpha=.22)
            axis.legend(loc='upper right', fontsize=7, ncol=3)
            self.position_cursors.append(axis.axvline(0, color='#de8a28', lw=1))
        self.position_canvas, self.position_toolbar = self.embed_figure(self.position_figure, self.position_tab)
        self.refresh_position_analysis()

    def refresh_position_analysis(self):
        if getattr(self, 'position_tab', None) is None:
            return
        i = self.star_index
        exact = self.position_analysis_arrays['analytic_positions'][:, i]
        actual = self.trajectories[i]
        series = {}
        for j, (num_line, ref_line) in enumerate(self.position_coordinate_lines):
            self.set_time_series(num_line, self.times, actual[:, j])
            self.set_time_series(ref_line, self.times, exact[:, j])
            series[f'{"xyz"[j]} numerical'] = actual[:, j]
            series[f'{"xyz"[j]} analytic'] = exact[:, j]
        axis = self.position_axes[0]
        axis.set_title(f'Star {i+1}: numerical (solid) vs analytic (dashed)', fontsize=9)
        self.register_hover(axis, self.times, series, title=f'Star {i+1} positions')
        axis.relim(); axis.autoscale_view(scalex=False)
        for axis, key in zip(self.position_axes[1:], ('position_error_norms','position_error_normalized')):
            if self.position_error_scale.get() == 'symlog':
                values = self.position_analysis_arrays[key]
                threshold = max(float(np.max(values))*1e-6, 1e-16)
                axis.set_yscale('symlog', linthresh=threshold)
            else:
                axis.set_yscale('linear')
            axis.relim(); axis.autoscale_view(scalex=False)
        self.position_canvas.draw_idle()
