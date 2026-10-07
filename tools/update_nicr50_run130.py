"""Use run 130's working PI and later indicated-temperature ramp bias.

No new R(T) calibration is inferred. Higher-temperature bias remains estimated.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tools.update_nicr50_run_profile import write_outputs


def derive_partial_table(data, diagnostics):
    if len(data) != len(diagnostics) or np.any(abs(data.time-diagnostics.time) >= .1):
        raise ValueError('CSV and diagnostics must align.')
    valid = diagnostics.status.eq('accepted')
    valid &= np.isfinite(data[['V','I','P','T','set_T']]).all(axis=1)
    valid &= np.isfinite(diagnostics[['raw_temperature_c','applied_current_a']]).all(axis=1)
    valid &= (data.I > 0) & (data.P > 0) & (diagnostics.applied_current_a > 0)
    valid &= (data.time-data.time.iloc[0]) >= 300
    valid &= abs(data['T']-data.set_T) <= 12
    valid &= abs(diagnostics.raw_temperature_c-data['T']) <= 8
    for i in np.flatnonzero(diagnostics.range_change_count.diff().fillna(0).to_numpy() > 0):
        valid.iloc[i:i+3] = False
    points, bins = [], []
    for low,high in [(120,140),(140,160),(160,180),(180,195),(195,205)]:
        mask = valid & diagnostics.raw_temperature_c.ge(low) & diagnostics.raw_temperature_c.lt(high)
        if mask.sum() < 20:
            continue
        points.append(dict(temperature_c=round(float(diagnostics.loc[mask,'raw_temperature_c'].mean()),6),
            current_a=round(float(np.sqrt(np.mean(diagnostics.loc[mask,'applied_current_a']**2))),9),
            wire_current_a=round(float(np.sqrt(np.mean(data.loc[mask,'I']**2))),9),
            measured_on_this_wire=True))
        bins.append(dict(temperature_bin_c=[low,high],samples=int(mask.sum()),
            median_tracking_error_c=float((data.loc[mask,'T']-data.loc[mask,'set_T']).median())))
    if len(points) < 4 or points[-1]['temperature_c'] < 195:
        raise ValueError('Insufficient later-ramp coverage around 120..200 C.')
    if any(b['current_a'] <= a['current_a'] for a,b in zip(points,points[1:])):
        raise ValueError('Partial ramp has a current reversal; review rather than silently fitting it.')
    return points,bins


def update_profile(profile, run):
    if profile.get('profile_name') != 'NiCr_50' or profile.get('parallel_resistance_ohm') != 20:
        raise ValueError('Expected the 20 Ohm NiCr_50 profile.')
    metadata = json.loads((run/'run_metadata.json').read_text())
    if metadata.get('parallel_resistance_ohm') != 20 or metadata.get('pid_gain_schedule'):
        raise ValueError('Expected run 130 with a 20 Ohm bank and flat PI gains.')
    if metadata.get('pid_kp') != .001 or not np.isclose(metadata.get('pid_ki',0),.001/600,rtol=1e-9,atol=0):
        raise ValueError('Unexpected PI gains; this revision requires the reviewed run-130 gains.')
    points,bins = derive_partial_table(pd.read_csv(run/'data.csv'),pd.read_json(run/'control_diagnostics.jsonl',lines=True))
    result = copy.deepcopy(profile)
    previous = result['current_feedforward_provenance'].get('run130_revision',{})
    baseline = copy.deepcopy(previous.get('previous_table',profile['current_feedforward_table']))
    anchor = points[-1]
    old_bias = float(np.interp(anchor['temperature_c'],[p['temperature_c'] for p in baseline],
                                              [p['current_a'] for p in baseline]))
    offset = anchor['current_a']-old_bias
    if not 0 < offset < .04:
        raise ValueError('High-temperature bias offset is outside the reviewed 0..40 mA range.')
    table = [p for p in baseline if p['temperature_c'] < 120] + points
    for point in baseline:
        if point['temperature_c'] <= anchor['temperature_c']:
            continue
        estimated = copy.deepcopy(point)
        estimated['current_a'] = round(point['current_a']+offset,9)
        estimated['wire_current_a'] = round(point['wire_current_a']*estimated['current_a']/point['current_a'],9)
        estimated['measured_on_this_wire'] = False
        table.append(estimated)
    # Join into the retained higher-T shape with a rising power bias, rather
    # than reintroducing the old near-flat 200..230 C section after the splice.
    right = min((p for p in table if p['temperature_c'] >= 240), key=lambda p:p['temperature_c'])
    for point in table:
        if anchor['temperature_c'] < point['temperature_c'] < right['temperature_c']:
            ratio = point['wire_current_a']/point['current_a']
            f = (point['temperature_c']-anchor['temperature_c'])/(right['temperature_c']-anchor['temperature_c'])
            point['current_a'] = round(float(np.sqrt(anchor['current_a']**2 + f*(right['current_a']**2-anchor['current_a']**2))),9)
            point['wire_current_a'] = round(point['current_a']*ratio,9)
    if any(b['current_a'] < a['current_a'] for a,b in zip(table,table[1:])):
        raise ValueError('The joined table must remain monotonic.')
    result.update(current_feedforward_table=table,pid_kp=.001,pid_ki=.001/600,
        pid_integral_time_s=600.,pid_gain_schedule=[],pid_integral_current_limit_a=3.,
        max_current=3.,max_wire_current_a=3.,max_power_w=15.,max_sample_voltage=5.,compliance_voltage=5.,
        t0_calibration_samples=9)
    provenance = result['current_feedforward_provenance']
    provenance.update(source_run='130_test',previous_source_run='127_test',
        method='Run-130 later-ramp RMS applied commands against raw indicated T; retain low-T points and estimated higher-T shape from run 127/129 with a constant command offset.',
        derived_temperature_range_c=[points[0]['temperature_c'],anchor['temperature_c']],
        bins=bins,startup_exclusion_s=300,
        measured_indicated_temperature_max_c=float(pd.read_json(run/'control_diagnostics.jsonl',lines=True).raw_temperature_c.max()),
        unmeasured_extension=dict(range_c=[anchor['temperature_c'],600.],measured=False,
            method='Retained higher-temperature shape shifted by the run-130 terminal bias offset; rising I squared join to 249 C. No new higher-temperature measurement.'),
        measured_final_hold=False,
        limitations='Provisional indicated-temperature ramp bias, not independent physical thermometry. Run 130 T0 scatter was 82.37 C; only about 120..200 C is newly sampled. The extension above this is an engineering estimate; R(T) remains extrapolated above 293.4 C.')
    provenance['run130_revision'] = dict(source_run='130_test',previous_table=baseline,bins=bins,
        measured_indicated_range_c=[points[0]['temperature_c'],anchor['temperature_c']],
        high_temperature_command_offset_a=offset,
        extension='Constant command offset above final run-130 bin, preserving previous shape and a rising I squared join to 249 C; all adjusted higher-T points are estimates.',
        source_temperature_max_c=float(pd.read_csv(run/'data.csv')['T'].max()),
        t0_spread_c=82.37,independent_temperature_calibration=False,
        selection='Accepted readings after 300 s, indicated error within 12 C, raw/filter difference within 8 C; omit range changes and next two cycles; minimum 20 per bin.',
        pi='Use the exact flat Kp/Ki/Ti from run 130, no gain schedule; correct bias rather than guessing stronger gains.',
        limit_authorization='User explicitly requested 5 V, 3 A and 15 W after run 130; both total and raw wire-current ceilings are 3 A, with no hidden old 30 mA wire cap. These settings are not verified wire ratings.',
        hashes={n:hashlib.sha256((run/n).read_bytes()).hexdigest() for n in ['data.csv','control_diagnostics.jsonl','run_metadata.json']})
    provenance['electrical_limits'] = dict(max_total_psu_current_a=3.,max_wire_current_a=3.,
        max_power_w=15.,max_sample_voltage_v=5.,compliance_voltage_v=5.)
    divider = provenance['parallel_divider']
    divider.update(resistor_max_power_at_compliance_w=1.25,
        wire_current_units='Measured wire-branch amperes for retained/observed ramp points; adjusted higher-T entries are estimates.',
        limitations='5 V across this bank is 1.25 W, equal to its nominal rating with no margin. Use an adequately cooled higher-rated 20 Ohm bank for sustained operation near 5 V. A 3 A command ceiling does not make 3 A attainable through the present load at 5 V.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--profiles',type=Path,default=Path('files/material_profiles'))
    args = parser.parse_args()
    result = update_profile(json.loads((args.profiles/'NiCr_50.json').read_text()),args.run)
    write_outputs(result,args.profiles)
    print('Updated NiCr_50 from run 130:',result['current_feedforward_provenance']['derived_temperature_range_c'],
          '600 C estimated command:',result['current_feedforward_table'][-1]['current_a'])


if __name__ == '__main__':
    main()
