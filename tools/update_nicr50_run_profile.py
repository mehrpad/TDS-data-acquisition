"""Derive a thin-NiCr ramp bias from accepted wire-branch measurements.

The temperature axis is indicated temperature, not independently measured T.
Applied total PSU commands are paired with the measurement they produced.
No instrument access and no changes to the source measurements or R(T) curve.
"""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def monotonic_squared_currents(currents, weights):
    """Weighted least-squares isotonic fit in heating-power units (I squared)."""
    blocks = []
    for i, (current, weight) in enumerate(zip(currents, weights)):
        blocks.append([current**2, weight, i, i+1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            a, b = blocks[-2:]
            weight = a[1]+b[1]
            blocks[-2:] = [[(a[0]*a[1]+b[0]*b[1])/weight, weight, a[2], b[3]]]
    fitted = np.empty(len(currents))
    for value, _, start, end in blocks:
        fitted[start:end] = np.sqrt(value)
    return fitted


def derive_table(data, diagnostics):
    if len(data) != len(diagnostics) or np.any(abs(data.time-diagnostics.time) >= .1):
        raise ValueError('CSV and diagnostics must align before fitting the current table.')
    valid = diagnostics.status.eq('accepted')
    valid &= np.isfinite(data[['V','I','P']]).all(axis=1)
    valid &= np.isfinite(diagnostics[['raw_temperature_c','applied_current_a']]).all(axis=1)
    valid &= (data.I > 0) & (data.P > 0) & (diagnostics.applied_current_a > 0)
    valid &= (data.time-data.time.iloc[0]) >= 180
    # Omit range-transition cycles and the following two cycles from calibration.
    for i in np.flatnonzero(diagnostics.range_change_count.diff().fillna(0).to_numpy() > 0):
        valid.iloc[i:i+3] = False
    points, bins = [], []
    for low in range(60, 600, 20):
        mask = valid & diagnostics.raw_temperature_c.ge(low) & diagnostics.raw_temperature_c.lt(low+20)
        if mask.sum() < 12:
            continue
        temperature = float(diagnostics.loc[mask,'raw_temperature_c'].mean())
        current = float(np.sqrt(np.mean(diagnostics.loc[mask,'applied_current_a']**2)))
        wire = float(np.sqrt(np.mean(data.loc[mask,'I']**2)))
        points.append(dict(temperature_c=round(temperature,6), current_a=current,
                           wire_current_a=round(wire,9), measured_on_this_wire=True))
        bins.append(dict(temperature_bin_c=[low,low+20], samples=int(mask.sum()),
                         raw_rms_psu_command_a=current, rms_wire_current_a=wire,
                         mean_tracking_error_c=float((data.loc[mask,'T']-data.loc[mask,'set_T']).mean())))
    if len(points) < 10 or points[-1]['temperature_c'] < 580:
        raise ValueError('Insufficient accepted ramp coverage to extend this table to 600 C.')
    currents = np.array([p['current_a'] for p in points])
    fitted = monotonic_squared_currents(currents, [b['samples'] for b in bins])
    if np.max(abs(fitted-currents)) > .003:
        raise ValueError('Current reversals exceed the allowed 3 mA fitting correction; review the run.')
    for point, current in zip(points,fitted):
        point['current_a'] = round(float(current),9)
    # Only the small remaining interval to 600 is estimated; do not claim a hold.
    fit = points[-4:]
    slope = float(np.polyfit([p['temperature_c'] for p in fit], [p['current_a']**2 for p in fit],1)[0])
    wire_slope = float(np.polyfit([p['temperature_c'] for p in fit], [p['wire_current_a']**2 for p in fit],1)[0])
    if slope <= 0 or wire_slope <= 0:
        raise ValueError('Endpoint extrapolation requires positive power slopes.')
    delta = 600-points[-1]['temperature_c']
    endpoint = dict(temperature_c=600., current_a=round(float(np.sqrt(points[-1]['current_a']**2+slope*delta)),9),
                    wire_current_a=round(float(np.sqrt(points[-1]['wire_current_a']**2+wire_slope*delta)),9),
                    measured_on_this_wire=False)
    table = [dict(temperature_c=23.,current_a=.001,measured_on_this_wire=False)] + points + [endpoint]
    evidence = dict(bins=bins, startup_exclusion_s=180,
        selection='Accepted finite positive samples; omit first 180 s and range-change cycle plus next two cycles; bin raw indicated T, use RMS applied total command from diagnostics, not next accepted command.',
        derived_temperature_range_c=[points[0]['temperature_c'],points[-1]['temperature_c']],
        maximum_monotonic_correction_a=float(np.max(abs(fitted-currents))),
        measured_indicated_temperature_max_c=float(diagnostics.loc[valid,'raw_temperature_c'].max()),
        measured_final_hold=False,
        unmeasured_extension=dict(range_c=[points[-1]['temperature_c'],600.], measured=False,
            method='Positive I squared versus indicated T fit to last four accepted ramp bins; anchored to final bin.',
            slope_a2_per_c=slope, fit_range_c=[fit[0]['temperature_c'],fit[-1]['temperature_c']]))
    return table,evidence


def update_profile(profile, run):
    if profile.get('profile_name') != 'NiCr_50':
        raise ValueError('Only the thin NiCr_50 profile may be updated.')
    metadata = json.loads((run/'run_metadata.json').read_text())
    program = metadata['experiment_program']
    if metadata['experiment_mode'] != 'TEMPERATURE' or len(program) != 1 or program[0]['ramp_speed_min'] != 10:
        raise ValueError('Expected the single 10 C/min temperature ramp.')
    table,evidence = derive_table(pd.read_csv(run/'data.csv'), pd.read_json(run/'control_diagnostics.jsonl',lines=True))
    result = copy.deepcopy(profile)
    result.update(current_feedforward_table=table, parallel_resistance_ohm=20.,
                  max_current=.18, pid_kp=.0000825, pid_ki=.00000055, pid_integral_time_s=150.,
                  pid_integral_current_limit_a=.02, temperature_prediction_time_s=5.,
                  t0_pair_samples=15, t0_dmm_current_range_a=.002, dmm_current_range_a=.002)
    if max(p['current_a'] for p in table) >= .95*result['max_current']:
        raise ValueError('Fitted table exceeds total PSU command headroom.')
    if table[-1]['wire_current_a'] >= .95*result['max_wire_current_a']:
        raise ValueError('Fitted table exceeds measured wire-current guard headroom.')
    source_bounds = json.loads((run/'curve_metadata.json').read_text())['source_temperature_bounds_c']
    hashes = {name.replace('.','_')+'_sha256':hashlib.sha256((run/name).read_bytes()).hexdigest()
              for name in ('data.csv','control_diagnostics.jsonl','run_metadata.json','r_vs_t_source.csv')}
    previous = result['current_feedforward_provenance']
    result['current_feedforward_provenance'] = dict(
        kind='provisional_ramp', ramp_rate_c_min=10., source_run='127_test',
        method='RMS observed applied total PSU commands versus raw indicated-temperature bins from run 127, with a weighted monotonic power fit.',
        measured_on_this_wire=True, independent_temperature_calibration=False,
        source_curve_temperature_bounds_c=source_bounds, wire_diameter_um=50., wire_length_known=False,
        reference_curve_file=previous['reference_curve_file'],
        startup_anchor='23 C / 1 mA total PSU retained; nominal wire current about 0.19 mA at 85.6 Ohm; not measured equilibrium.',
        limitations='Observed ramp commands only, no measured equilibrium hold. Temperature is inferred from the existing donor R(T), extrapolated above 293.4 C. T0 scatter was 7.32 C. Correcting this current table does not independently validate physical temperature.',
        parallel_divider=dict(parallel_resistance_ohm=20., nominal_wire_resistance_ohm=85.5984035007845,
            nominal_total_to_wire_current_ratio=1+85.5984035007845/20.,
            table_current_units='total PSU amperes', wire_current_units='measured wire-branch amperes; endpoint is extrapolated',
            resistor_minimum_power_rating_w=1.25, resistor_max_power_at_compliance_w=.45,
            resistor_bank='Five 100 Ohm / 0.25 W resistors in parallel; 20 Ohm / 1.25 W nominal.',
            method='Table uses observed total commands for the actual user-reported 20 Ohm bank; no fixed divider ratio is applied to run data.',
            required_wiring='20 Ohm bank outside chamber across PSU output terminals; branch ammeter only, Kelvin voltage only across wire, PSU sense at PSU terminals.',
            limitations='PSU commands are not independent measurements of total current; ammeter burden, resistor tolerance and mounting remain part of this calibration.'),
        electrical_limits=dict(max_total_psu_current_a=.18,max_wire_current_a=result['max_wire_current_a'],
            max_power_w=result['max_power_w'], max_sample_voltage_v=result['max_sample_voltage'],
            compliance_voltage_v=result['compliance_voltage']),
        run127_review=dict(saved_cycles=1698, rejected_cycles=1, t0_temperature_spread_c=7.32,
            recorded_parallel_resistance_ohm=metadata['parallel_resistance_ohm'], actual_parallel_resistance_ohm=20.,
            maximum_filtered_tracking_error_c=66.7489665039447,
            shutdown='Raw indicated T 600.023139 C exceeded the 600 C limit; V=2.43919092 V, wire I=0.0267111764 A, R=91.317240524 Ohm. No open-circuit signature in saved terminal measurement.'),
        **evidence, **hashes)
    return result


def write_outputs(profile, directory):
    (directory/'NiCr_50.json').write_text(json.dumps(profile,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    with (directory/'NiCr_50_current_table.csv').open('w',newline='',encoding='utf-8') as stream:
        writer = csv.DictWriter(stream,fieldnames=['temperature_c','current_a','wire_current_a','measured_on_this_wire'])
        writer.writeheader();writer.writerows(profile['current_feedforward_table'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--profiles',type=Path,default=Path('files/material_profiles'))
    args=parser.parse_args()
    profile=update_profile(json.loads((args.profiles/'NiCr_50.json').read_text()),args.run)
    write_outputs(profile,args.profiles)
    print('Updated NiCr_50 for five parallel 100 Ohm resistors:',len(profile['current_feedforward_table']),'points;',profile['current_feedforward_table'][-1])


if __name__=='__main__':
    main()
