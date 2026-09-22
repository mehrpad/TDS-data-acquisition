"""Build a provisional Ni bias from reviewed current-ramp data; preserve safety limits."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def update_profile(profile, data, excluded_rows=()):
    """Preserve the old interpolation through 100 C and blend over 100-150 C.

    CSV row indices are zero-based data rows. Missing T is never reconstructed
    from rejected resistance readings. This is not an equilibrium calibration.
    """
    old = profile['current_feedforward_table']
    temperatures = np.array([p['temperature_c'] for p in old])
    currents = np.array([p['current_a'] for p in old])
    if not np.all(np.diff(temperatures) > 0) or not temperatures[0] <= 100 <= temperatures[-1]:
        raise ValueError('Base table must be ordered and cover 100 C.')
    anchor = float(np.interp(100., temperatures, currents))
    finite = np.isfinite(data[['time', 'T', 'I', 'C_V', 'P']]).all(axis=1)
    keep = finite & (data['T'] > 100) & (data['T'] <= 600) & (data.I > 0) & (data.C_V > 0) & (data.P > 0)
    keep &= ~data.index.isin(excluded_rows)
    selected = data[keep].copy()
    if len(selected) < 6:
        raise ValueError('Insufficient valid hot-ramp measurements.')
    selected['bin'] = np.floor(selected['T'] / 25)
    points, evidence = [], []
    for _, group in selected.groupby('bin', sort=True):
        t = float(group['T'].mean())
        current = float(np.sqrt(np.mean(group.C_V ** 2)))
        points.append((t, current))
        evidence.append({'temperature_c': t, 'rms_command_current_a': current,
                         'csv_data_rows': [int(i) for i in group.index], 'samples': len(group)})
    t, current = np.array(points).T
    if np.any(np.diff(current) < 0):
        raise ValueError('Binned current is non-monotonic; review data explicitly.')
    result = copy.deepcopy(profile)
    table = [copy.deepcopy(p) for p in old if p['temperature_c'] < 100]
    table.append({'temperature_c': 100., 'current_a': anchor})
    for target in sorted(set(t.tolist() + [150.])):
        weight = np.clip((target - 100.) / 50., 0., 1.)
        bias = anchor + weight * (float(np.interp(target, t, current)) - anchor)
        table.append({'temperature_c': round(target, 6), 'current_a': round(bias, 9)})
    if table[-1]['temperature_c'] < 600:
        table.append({'temperature_c': 600., 'current_a': table[-1]['current_a']})
    result['current_feedforward_table'] = table
    provenance = result['current_feedforward_provenance']
    provenance.pop('unmeasured_extension', None)
    provenance['limitations'] = ('Mixed provisional ramp bias, not equilibrium calibration or validated tuning. '
        'The fast current ramp may overestimate current for a 10 C/min temperature ramp. '
        'PI must correct thermal lag; temperature above the source R(T) range remains extrapolated. '
        'Electrical limits are preserved by this updater; table currents are clamped by the controller.')
    provenance['current_ramp_update'] = {
        'preserved_range_c': [float(temperatures[0]), 100.],
        'valid_temperature_range_c': [float(selected['T'].min()), float(selected['T'].max())],
        'method': '25 C bins: mean valid logged T and RMS applied command C_V; no NaN reconstruction',
        'transition_range_c': [100., 150.],
        'transition_method': 'Linear blend from the old 100 C bias to the interpolated current-ramp bias',
        'excluded_csv_data_rows': list(excluded_rows),
        'input_rows': len(data), 'used_rows': len(selected),
        'interpolation_gaps_c': [[float(a), float(b)] for a,b in zip(np.sort(selected['T'])[:-1], np.sort(selected['T'])[1:]) if b-a > 50],
        'bins': evidence,
        'upper_placeholder': {'temperature_c': 600., 'method': 'Clamp last binned current; not a measured 600 C point'},
    }
    prior_limits = profile.get('current_feedforward_provenance', {}).get('current_ramp_update', {}).get('electrical_limits')
    if prior_limits:
        provenance['current_ramp_update']['electrical_limits'] = copy.deepcopy(prior_limits)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads((args.run/'run_metadata.json').read_text(encoding='utf-8'))
    if metadata.get('experiment_mode') != 'CURRENT' or metadata.get('current_feedforward_provenance', {}).get('run') != '139_Ni100_uncharged_test_1':
        raise ValueError('Expected the reviewed Ni current-ramp configuration.')
    data = pd.read_csv(args.run/'data.csv')
    # Row 55 jumps 484->578 C, followed by six invalid temperatures, then
    # returns to ~576 C at substantially higher current. Explicitly exclude it.
    if args.run.name != '107_test' or len(data) != 89 or not np.isclose(data.iloc[55]['T'], 578.450928, atol=.001):
        raise ValueError('Exclusion is specific to reviewed run 107; review a new run explicitly.')
    profile = json.loads(args.profile.read_text(encoding='utf-8'))
    if profile.get('profile_name') != 'Ni_100_152':
        raise ValueError('Run 107 cannot update a different material profile.')
    result = update_profile(profile, data, excluded_rows=(55,))
    source = result['current_feedforward_provenance']['current_ramp_update']
    source.update(run=args.run.name, data_sha256=hashlib.sha256((args.run/'data.csv').read_bytes()).hexdigest(),
                  current_ramp_a_min=metadata['experiment_program']['ramp_speed_min'],
                  exclusion_reason='Row 55 isolated upward spike followed by six invalid T rows and return to lower T at higher current')
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    print(json.dumps(result['current_feedforward_table'], indent=2))


if __name__ == '__main__':
    main()
