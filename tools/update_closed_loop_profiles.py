"""Rebuild provisional wire biases from the reviewed 10 C/min runs 108 and 109.

The temperature axis is the logged indicated T, not the requested setpoint.
No instrument access. Electrical limits, gains and source run files are preserved.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = [('108_Ni_100_hydrogen_cur_3_10k_min', 'Ni_100_152', False),
        ('109_NiCr_uncharged_test', 'NiCr_100_163', True)]


def derive_table(data, diagnostics, nicr=False):
    if len(data) != len(diagnostics) or not np.all(np.abs(data.time - diagnostics.time) < .1):
        raise ValueError('CSV and diagnostics must align before selecting accepted measurements.')
    elapsed = data.time - data.time.iloc[0]
    valid = np.isfinite(data[['time', 'set_T', 'T', 'I', 'C_V', 'P']]).all(axis=1)
    valid &= (diagnostics.status == 'accepted') & (data.I > 0) & (data.C_V > 0) & (data.P > 0)
    valid &= elapsed >= (180 if nicr else 60)
    # Meter-settling transients are not representative operating points.
    for i in np.flatnonzero(diagnostics.range_change_count.diff().fillna(0).to_numpy() > 0):
        valid.iloc[i:i+3] = False
    target = float(data.set_T.max())
    ramp = valid & (data.set_T < target - 1e-6)
    edges = np.arange(60, 501, 20) if nicr else np.r_[np.arange(30, 100, 10), np.arange(100, 601, 20)]
    points, bins = [], []
    for low, high in zip(edges[:-1], edges[1:]):
        group = data[ramp & (data['T'] >= low) & (data['T'] < high)]
        if len(group) < 12:
            continue
        t = float(group['T'].mean())
        current = float(np.sqrt(np.mean(group.C_V ** 2)))
        points.append({'temperature_c': round(t, 6), 'current_a': round(current, 9)})
        bins.append({'temperature_bin_c': [float(low), float(high)], 'samples': len(group),
                     'mean_indicated_temperature_c': round(t, 6), 'rms_command_current_a': round(current, 9),
                     'mean_tracking_error_c': round(float((group['T']-group.set_T).mean()), 6)})
    if len(points) < 8 or points[-1]['temperature_c'] < target - 50:
        raise ValueError('Insufficient ramp coverage near the target.')
    # Small quantization-scale reversals can occur in the lowest bins.
    raw = np.array([p['current_a'] for p in points])
    monotone = np.maximum.accumulate(raw)
    correction = float(np.max(monotone - raw))
    if correction > .001:
        raise ValueError('More than one current step of monotonic correction; review the run.')
    for point, current in zip(points, monotone):
        point['current_a'] = float(current)
    tail = data[valid & (data.set_T == target) & (data.time >= data.time.iloc[-1]-60)]
    if len(tail) < 20 or tail.time.iloc[-1] - tail.time.iloc[0] < 40:
        raise ValueError('Need at least 40 seconds of accepted final-target readings.')
    tail_t = float(tail['T'].mean())
    tail_i = float(np.sqrt(np.mean(tail.C_V ** 2)))
    if tail_t <= points[-1]['temperature_c'] or tail_i < points[-1]['current_a']:
        raise ValueError('Late-hold point does not join the ramp monotonically.')
    # Explicitly estimated extension: fit I^2 against T over the last four ramp bins,
    # anchored to the observed late-hold current. Do not claim high-T measurements.
    fit = points[-4:]
    slope = float(np.polyfit([p['temperature_c'] for p in fit], [p['current_a']**2 for p in fit], 1)[0])
    if not np.isfinite(slope) or slope <= 0 or tail_t >= 600:
        raise ValueError('Cannot construct a positive bounded extension to 600 C.')
    table = [{'temperature_c': 23., 'current_a': .01}] + points
    table.append({'temperature_c': round(tail_t, 6), 'current_a': round(tail_i, 9)})
    endpoint = float(np.sqrt(tail_i**2 + slope * (600-tail_t)))
    table.append({'temperature_c': 600., 'current_a': round(endpoint, 9)})
    evidence = {
        'bins': bins, 'monotonic_current_correction_a': correction,
        'startup_exclusion_s': 180 if nicr else 60,
        'selection': 'Accepted finite positive samples; exclude first startup interval and range-change cycle plus next two cycles; ramp bins exclude final-target dwell',
        'derived_temperature_range_c': [points[0]['temperature_c'], round(tail_t, 6)],
        'late_hold': {'samples': len(tail), 'mean_temperature_c': round(tail_t, 6),
                      'rms_current_a': round(tail_i, 9),
                      'temperature_rate_c_min': round(float(np.polyfit(tail.time-tail.time.iloc[0], tail['T'], 1)[0]*60), 6),
                      'note': 'Observed last-minute bias, not proven equilibrium'},
        'unmeasured_extension': {'range_c': [round(tail_t, 6), 600.], 'measured': False,
            'method': 'Linear I squared versus indicated T fitted to last four ramp bins, anchored at late-hold mean',
            'slope_a2_per_c': slope, 'fit_range_c': [fit[0]['temperature_c'], fit[-1]['temperature_c']]},
    }
    return table, evidence


def update_profile(profile, run, nicr=False):
    metadata = json.loads((run/'run_metadata.json').read_text(encoding='utf-8'))
    expected = '136_NiCr_100_hydrogen_cur_3_10k_min' if nicr else '139_Ni100_uncharged_test_1'
    if metadata.get('current_feedforward_provenance', {}).get('run') != expected:
        raise ValueError('Source metadata does not match the expected material lineage.')
    program = metadata.get('experiment_program', [])
    if metadata.get('experiment_mode') != 'TEMPERATURE' or len(program) != 1 or program[0]['ramp_speed_min'] != 10:
        raise ValueError('Expected a single 10 C/min temperature program.')
    data = pd.read_csv(run/'data.csv')
    diagnostics = pd.read_json(run/'control_diagnostics.jsonl', lines=True)
    table, evidence = derive_table(data, diagnostics, nicr)
    result = copy.deepcopy(profile)
    if max(p['current_a'] for p in table) >= .95 * result['max_current']:
        raise ValueError('Derived table would exceed the existing measured-current guard.')
    result['current_feedforward_table'] = table
    provenance = {'kind': 'provisional_ramp', 'ramp_rate_c_min': 10., 'run': run.name,
                  'method': 'RMS applied command per indicated-temperature bin in the accepted closed-loop ramp',
                  'startup_anchor': '23 C / 10 mA retained as startup command, not a measured equilibrium point',
                  'limitations': 'Closed-loop ramp bias, not independently calibrated equilibrium current. Displayed T above source R(T) bounds is extrapolated.' + (' NiCr startup calibration scatter remains unresolved.' if nicr else ''),
                  **evidence}
    for name in ('data.csv', 'control_diagnostics.jsonl', 'run_metadata.json', 'r_vs_t_source.csv'):
        provenance[name.replace('.', '_')+'_sha256'] = hashlib.sha256((run/name).read_bytes()).hexdigest()
    provenance['source_curve_temperature_bounds_c'] = json.loads((run/'curve_metadata.json').read_text())['source_temperature_bounds_c']
    provenance['electrical_limits'] = profile.get('current_feedforward_provenance', {}).get('current_ramp_update', {}).get('electrical_limits',
        profile.get('current_feedforward_provenance', {}).get('electrical_limits', {'method': 'Retained existing profile electrical limits'}))
    result['current_feedforward_provenance'] = provenance
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--profiles', type=Path, default=Path('files/material_profiles'))
    args = parser.parse_args()
    updates = []
    for run_name, name, nicr in RUNS:
        path = args.profiles/(name+'.json')
        updated = update_profile(json.loads(path.read_text(encoding='utf-8')), args.source/run_name, nicr)
        updates.append((path, updated))
    for path, profile in updates:
        path.write_text(json.dumps(profile, indent=2, sort_keys=True)+'\n', encoding='utf-8')
        print(path.name, len(profile['current_feedforward_table']), 'points; endpoint',profile['current_feedforward_table'][-1])


if __name__ == '__main__':
    main()
