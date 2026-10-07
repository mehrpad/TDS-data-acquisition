"""Revise the thin NiCr controller after run 129 without fitting its noisy T0.

Run after update_nicr50_run_profile.py when rebuilding the run-127 baseline.
The adjusted table points are engineering estimates, not new measurements.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from tools.update_nicr50_run_profile import write_outputs


def retune_profile(profile):
    if profile.get('profile_name') != 'NiCr_50' or profile.get('parallel_resistance_ohm') != 20:
        raise ValueError('Expected NiCr_50 with the 20 Ohm parallel resistor bank.')
    result = copy.deepcopy(profile)
    provenance = result['current_feedforward_provenance']
    if provenance.get('source_run') != '127_test':
        raise ValueError('Expected the run-127 current-table baseline.')
    table = result['current_feedforward_table']
    # Restore the original observations before a repeat invocation.
    original = provenance.get('run129_controller_revision', {}).get('original_points', [])
    originals = {p['temperature_c']: p for p in original}
    for i, point in enumerate(table):
        if point['temperature_c'] in originals:
            table[i] = copy.deepcopy(originals[point['temperature_c']])
    left = max((p for p in table if p['temperature_c'] <= 340), key=lambda p:p['temperature_c'])
    right = min((p for p in table if p['temperature_c'] >= 420), key=lambda p:p['temperature_c'])
    if right['current_a'] <= left['current_a']:
        raise ValueError('Table smoothing requires increasing boundary currents.')
    original = []
    for point in table:
        if left['temperature_c'] < point['temperature_c'] < right['temperature_c']:
            original.append(copy.deepcopy(point))
            fraction = (point['temperature_c']-left['temperature_c'])/(right['temperature_c']-left['temperature_c'])
            estimate = float(np.sqrt(left['current_a']**2 + fraction*(right['current_a']**2-left['current_a']**2)))
            # Retain the observed local divider ratio for an explicitly estimated
            # wire-current column. Do not relabel estimates as measured currents.
            wire_ratio = point['wire_current_a']/point['current_a']
            point.update(current_a=round(estimate,9), wire_current_a=round(estimate*wire_ratio,9),
                         measured_on_this_wire=False)
    result.update(pid_kp=.000165, pid_ki=.00000165, pid_integral_time_s=100.,
        pid_integral_current_limit_a=result['max_current'], current_quantization_hysteresis_a=.0001,
        pid_gain_schedule=[dict(current_a=.04,kp=.0000825,ki=.00000055,kd=0.),
                           dict(current_a=.07,kp=.000165,ki=.00000165,kd=0.)])
    provenance['parallel_divider']['wire_current_units'] = (
        'Wire-branch amperes; ramp observations except the estimated endpoint and smoothed bias points.')
    provenance['run129_controller_revision'] = dict(
        source_run='129_NiCr_50_hydrogen_curr_1', independent_temperature_calibration=False,
        run129_used_as_thermal_calibration=False, t0_spread_c=92.42,
        method='Engineering I squared interpolation between unchanged run-127 anchors removes the near-flat bias segment; original observations retained below.',
        smoothing_range_c=[left['temperature_c'],right['temperature_c']], original_points=original,
        gains='Retain startup gains below 40 mA table bias, blend to twice Kp and three times Ki at 70 mA; higher-current Ti=100 s.',
        integral_limit='Correction state bounded by the configured total PSU current ceiling; removes the separate 20 mA correction cap. Output anti-windup and all electrical/temperature guards remain.',
        current_hysteresis_a=.0001,
        retained_limits='180 mA total command, 30 mA measured wire, 0.10 W wire power, 3 V compliance and 600 C indicated temperature. Full PSU output is unsuitable for the 20 Ohm / 1.25 W bank.',
        limitations='Provisional engineering tuning, not a validated thermal model. Run 129 has 92.42 C T0 scatter; repeat cold calibration before judging physical temperature or fitting a new curve.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profiles',type=Path,default=Path('files/material_profiles'))
    args = parser.parse_args()
    profile = retune_profile(json.loads((args.profiles/'NiCr_50.json').read_text()))
    write_outputs(profile,args.profiles)
    print('Updated NiCr_50 table bias and scheduled PI; retained electrical guards.')


if __name__ == '__main__':
    main()
