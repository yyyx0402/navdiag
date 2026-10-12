# SPDX-License-Identifier: GPL-3.0-or-later
"""Full-drive positioning adapter; reference labels never enter SPP or risk features.

Standalone GPL tool using the pinned external RINEX helpers, not shipped in the MIT wheel.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyrtklib as prl

from prepare import DAYS, ROOT, extractor_module
from navdiag.positioning import ecef_to_enu, ecef_to_lla, lla_to_ecef
from navdiag.visibility import match_truth, read_urbannav_truth


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def prepare(raw_root, output, policy):
    output.mkdir(parents=True,exist_ok=True)
    extractor=extractor_module()
    inputs=[]
    for area,day in DAYS.items():
        directory=raw_root/area
        rover=list(directory.glob('*.ublox.f9p.splitter.obs'))
        if len(rover)!=1: raise ValueError(f'{area} requires one splitter observation file')
        paths=[rover[0],raw_root/'nav'/f'brdc{day}0.21n',
               raw_root/'nav'/f'HKSC00HKG_R_2021{day}0000_01D_CN.rnx',directory/'groundtruth.txt']
        inputs.extend({'path':path.relative_to(ROOT).as_posix(),'bytes':path.stat().st_size,'sha256':sha(path)} for path in paths)
    plan={'policy':policy,'source_files':inputs,'implementation_sha256':sha(Path(__file__)),
          'external_extractor_sha256':sha(ROOT/'.cache/klt_reference/extract_features.py')}
    frozen=output/'preprocessing_plan.json'
    if frozen.exists() and json.loads(frozen.read_text(encoding='utf-8'))!=plan:
        raise ValueError('Preprocessing plan changed; use a new output directory instead of overwriting evidence')
    frozen.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    frames,audits,event_frames,observation_frames=[],[],[],[]
    for area,day in DAYS.items():
        directory=raw_root/area
        rover=next(directory.glob('*.ublox.f9p.splitter.obs'))
        navfiles=[raw_root/'nav'/f'brdc{day}0.21n',raw_root/'nav'/f'HKSC00HKG_R_2021{day}0000_01D_CN.rnx']
        obs,nav=extractor.read_rinex(rover,navfiles)
        spans=extractor.epoch_slices(obs)
        raw_times=np.array([extractor.epoch_utc(obs.data[i].time) for i,_ in spans])
        spp_available=np.zeros(len(spans),dtype=bool)
        options=extractor.make_prcopt()
        options.elmin=np.radians(policy['elevation_mask_deg'])
        rows,dropped=[],Counter()
        previous,previous_time=None,None
        for raw_index,(i,m) in enumerate(spans):
            epoch=extractor.rover_epoch_obs(obs,i,m)
            if epoch.n<4:
                dropped['too_few_observations']+=1; continue
            sol=prl.sol_t(); sol.time=epoch.data[0].time
            ssat=prl.Arr1Dssat_t(prl.MAXSAT)
            azel=prl.Arr1Ddouble(epoch.n*2); message=prl.Arr1Dchar(128)
            succeeded=prl.pntpos(epoch.data.ptr,epoch.n,nav,options,sol,azel,ssat.ptr,message)
            xyz=np.array([sol.rr[k] for k in range(3)])
            if not succeeded or not np.isfinite(xyz).all() or np.linalg.norm(xyz)<1e6:
                dropped['spp_failed']+=1; continue
            used=[]
            for k in range(epoch.n):
                measurement=epoch.data[k]; status=ssat[measurement.sat-1]
                if status.vs and extractor.sat_name(measurement.sat)[0] in {'G','C'}:
                    used.append((float(status.azel[0]),float(status.azel[1]),float(status.resp[0]),
                                 float(measurement.SNR[0])/1000.))
            used=np.asarray(used)
            if len(used)<4 or not np.isfinite(used).all():
                dropped['invalid_used_satellites']+=1; continue
            used_azel=prl.Arr1Ddouble(len(used)*2)
            for k,value in enumerate(used[:,:2].ravel()): used_azel[k]=float(value)
            dop=prl.Arr1Ddouble(4); prl.dops(len(used),used_azel,options.elmin,dop)
            timestamp=int(round(extractor.epoch_utc(epoch.data[0].time)*1000))
            speed=0.
            if previous is not None and 0<timestamp-previous_time<5000:
                speed=min(70.,np.linalg.norm(xyz-previous)/((timestamp-previous_time)/1000))
            row={'timestamp_ms':timestamp,'satellite_count':len(used),'hdop':float(dop[2]),
                 'residual_rms_m':float(np.sqrt(np.mean(used[:,2]**2))),
                 'cn0_mean_dbhz':float(used[:,3].mean()),'cn0_min_dbhz':float(used[:,3].min()),
                 'speed_mps':float(speed),**{f'wls_{axis}_m':float(value) for axis,value in zip('xyz',xyz)}}
            if not np.isfinite(list(row.values())).all() or row['hdop']<=0:
                dropped['invalid_features']+=1; continue
            rows.append(row); spp_available[raw_index]=True; previous,previous_time=xyz,timestamp
        if not rows: raise ValueError(f'{area}: no valid SPP results')
        frame=pd.DataFrame(rows)
        # Coordinate origin and causal history do not depend on reference availability.
        origin=ecef_to_lla(frame[[f'wls_{axis}_m' for axis in 'xyz']].iloc[0].to_numpy())
        for axis,value in zip(['origin_lat_deg','origin_lon_deg','origin_alt_m'],origin): frame[axis]=float(value)
        frame[['wls_e_m','wls_n_m','wls_u_m']]=ecef_to_enu(frame[[f'wls_{axis}_m' for axis in 'xyz']].to_numpy(),origin)
        frame['trip_id']=f'UrbanNav_{area}_202105{int(day)-120:02d}'
        frame['device_id']='ublox_f9p_splitter'; frame['data_kind']='real'
        observation_frames.append(frame.copy())
        truth=read_urbannav_truth(directory/'groundtruth.txt')
        ids,matched=match_truth(frame.timestamp_ms.to_numpy()/1000,truth,policy['truth_time_tolerance_s'])
        quality=truth.quality.to_numpy()[ids]
        allowed=np.isin(quality,policy['truth_allowed_quality'])
        dropped['reference_time_unmatched']=int((~matched).sum())
        dropped['reference_quality_rejected']=int((matched & ~allowed).sum())
        keep=matched & allowed
        accepted=frame.loc[keep].copy().reset_index(drop=True)
        if accepted.empty: raise ValueError(f'{area}: no quality-accepted reference epochs')
        reference=truth.iloc[ids[keep]]
        truth_xyz=lla_to_ecef(reference[['latitude_deg','longitude_deg','height_m']].to_numpy())
        accepted[['gt_e_m','gt_n_m','gt_u_m']]=ecef_to_enu(truth_xyz,origin)
        accepted['horizontal_error_m']=np.linalg.norm(accepted[['wls_e_m','wls_n_m']].to_numpy()-accepted[['gt_e_m','gt_n_m']].to_numpy(),axis=1)
        raw_ids,raw_matched=match_truth(raw_times,truth,policy['truth_time_tolerance_s'])
        raw_quality=truth.quality.to_numpy()[raw_ids]
        raw_quality_allowed=np.isin(raw_quality,policy['truth_allowed_quality'])
        raw_eligible=raw_matched & raw_quality_allowed
        scorable=spp_available & raw_eligible
        events=pd.DataFrame({'trip_id':accepted.trip_id.iloc[0],'device_id':'ublox_f9p_splitter',
              'timestamp_ms':np.rint(raw_times*1000).astype('int64'),'solution_available':spp_available,
              'reference_time_matched':raw_matched,'reference_quality_allowed':raw_quality_allowed,
              'scorable':scorable})
        assert int(scorable.sum())==len(accepted)
        event_frames.append(events)
        frames.append(accepted)
        audit={'area':area,'trip_id':accepted.trip_id.iloc[0],'raw_epochs':len(spans),'spp_success_epochs':len(frame),
               'reference_accepted_epochs':len(accepted),'dropped':dict(dropped),'time_span_s':float((accepted.timestamp_ms.max()-accepted.timestamp_ms.min())/1000),
               'coverage':len(accepted)/len(spans),'raw_time_span_s':float(raw_times[-1]-raw_times[0]),
               'raw_reference_matched_epochs':int(raw_matched.sum()),'raw_reference_eligible_epochs':int(raw_eligible.sum()),
               'solution_availability':int(spp_available.sum())/len(spans),
               'solution_availability_on_eligible_reference':int(scorable.sum())/int(raw_eligible.sum()) if raw_eligible.sum() else None}
        audits.append(audit)
        print(json.dumps(audit,ensure_ascii=False),flush=True)
    table=pd.concat(frames,ignore_index=True)
    csv=output/'epochs.csv'; table.to_csv(csv,index=False)
    pd.concat(observation_frames,ignore_index=True).to_csv(output/'observations.csv',index=False)
    pd.concat(event_frames,ignore_index=True).to_csv(output/'availability.csv',index=False)
    report={'data_kind':'real','task':'epoch position error, not LOS/NLOS','baseline_solver':policy['solver'],
            'epochs':len(table),'trips':table.trip_id.nunique(),'devices':audits,'source_files':inputs,
            'dataset_sha256':sha(csv),'uses_raw_imu':False,
            'preprocessing':['GPS/BeiDou broadcast ephemeris, satellite clock and code bias via RTKLIB',
                             'Broadcast ionosphere and Saastamoinen troposphere corrections',
                             'SPP solver validity,10-degree elevation mask and used-satellite feature extraction',
                             'Independent SPAN time/quality guards; reference used only for labels and scoring'],
            'limitations':policy['limitations']}
    (output/'preprocess_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=Path,default=ROOT/'data/raw/urbannav')
    parser.add_argument('--output',type=Path,default=ROOT/'data/processed/urbannav_positioning_v3')
    parser.add_argument('--plan',type=Path,default=ROOT/'config/urbannav_positioning.json')
    args=parser.parse_args()
    prepare(args.raw,args.output,json.loads(args.plan.read_text(encoding='utf-8')))
