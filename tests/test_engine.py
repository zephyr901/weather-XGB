import copy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import threading
import unittest
import numpy as np
import pandas as pd

from weather_xgb.config import *
from weather_xgb.data import *
from weather_xgb.engine import *
from weather_xgb.examples import generate


class WeatherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.cfg = generate(self.root / "input")
        self.cfg.grid = [dict(n_estimators=4,max_depth=2,learning_rate=.1,min_child_weight=1,reg_lambda=1)]
        self.cfg.n_jobs = 1

    def tearDown(self):
        self.temp.cleanup()

    def test_roundtrip_and_arbitrary_scale(self):
        self.cfg.shapes[0].scale = "neighborhood"
        restored = from_dict(asdict(self.cfg))
        self.assertEqual(asdict(restored), asdict(self.cfg))
        jobs, audit = prepare_project(restored)
        self.assertEqual(len(jobs), 6)
        self.assertTrue(all(r['status'] == 'ready' for r in audit['reports']))
        self.assertTrue(all(j.scale == "neighborhood" for j in jobs))

    def test_target_missing_independent(self):
        v = self.cfg.variables[0]
        frame = pd.read_csv(v.source.path)
        frame.loc[frame.station_id == 'S00', 'temperature'] = np.nan
        frame.to_csv(v.source.path,index=False)
        jobs, _ = prepare_project(self.cfg)
        temp = next(j for j in jobs if j.variable_index == 0)
        wind = next(j for j in jobs if j.variable_index == 1)
        self.assertNotIn('S00',set(temp.data.station_id))
        self.assertIn('S00',set(wind.data.station_id))

    def test_coverage_leap_and_partial_month(self):
        times = pd.date_range('2024-02-01', '2024-03-01',freq='h',inclusive='left')
        raw = pd.DataFrame(dict(station_id='A',date=times,y=1.))
        p = ExperimentPreset(grain='monthly')
        v = VariableSpec(aggregation='sum')
        result, audit = aggregate(raw,v,p)
        self.assertEqual(result.y.iloc[0],696)
        self.assertEqual(audit.expected.iloc[0],696)
        result,_ = aggregate(raw.iloc[:100],v,p)
        self.assertTrue(result.y.isna().all())

    def test_no_upsampling_and_duplicate(self):
        raw = pd.DataFrame(dict(station_id=['A'],date=[pd.Timestamp('2025-01-01')],y=[1]))
        with self.assertRaisesRegex(ValueError,'Lower-frequency'): aggregate(raw,VariableSpec(frequency='D'),ExperimentPreset())
        source = self.cfg.variables[0].source
        frame = pd.read_csv(source.path)
        pd.concat([frame,frame.iloc[:1]]).to_csv(source.path,index=False)
        with self.assertRaisesRegex(ValueError,'duplicate'):load_variable(self.cfg.variables[0])

    def test_season_and_cross_midnight(self):
        p = ExperimentPreset(grain='seasonal')
        start,end,_ = period_bounds(pd.Timestamp('2025-01-09'),p)
        self.assertEqual(start,pd.Timestamp('2024-12-01'));self.assertEqual(end,pd.Timestamp('2025-03-01'))
        p = ExperimentPreset(grain='daynight',day_start=18,day_end=6)
        start,end,segment = period_bounds(pd.Timestamp('2025-01-02 02:00'),p)
        self.assertEqual(start,pd.Timestamp('2025-01-01 18:00'));self.assertEqual(segment,'day');self.assertEqual(end,pd.Timestamp('2025-01-02 06:00'))

    def test_forecast_gaps_and_no_future_leakage(self):
        dates = pd.date_range('2025-01-01',periods=40,freq='D')
        raw = pd.DataFrame(dict(station_id='A',date=dates,y=np.arange(40.),segment='all'))
        d,lags = forecast_features(raw,'daily')
        altered = raw.copy(); altered.loc[altered.date >= dates[25], 'y'] += 1000
        e,_ = forecast_features(altered,'daily',lags)
        cols=[c for c in d if c.startswith('h::')]
        pd.testing.assert_frame_equal(d.loc[d.date<=dates[25],cols],e.loc[e.date<=dates[25],cols])
        gap = raw[raw.date != dates[20]]
        g,_ = forecast_features(gap,'daily',[1])
        self.assertTrue(g.loc[g.date==dates[21],'h::lag_1'].isna().all())

    def test_station_and_time_splits(self):
        jobs,_=prepare_project(self.cfg)
        for j in jobs:
            task=self.cfg.presets[j.preset_index].task
            for a,b in splits(j.data,self.cfg,task):
                train,test=j.data.iloc[a],j.data.iloc[b]
                if task=='association':self.assertFalse(set(train.station_id)&set(test.station_id))
                else:self.assertLess(train.date.max(),test.date.min())

    def test_category_fit_training_only(self):
        frame=pd.DataFrame({'category':['A','A','B'],'numeric':[1,np.nan,3]})
        enc=Encoder(['category','numeric'],['category']).fit(frame.iloc[:2])
        out=enc.transform(frame.iloc[2:])
        self.assertTrue(np.isnan(out.iloc[0,0]));self.assertEqual(enc.categories,{'category':['A']})

    def test_background_effect_and_exclusion(self):
        jobs,_=prepare_project(self.cfg)
        job=next(j for j in jobs if self.cfg.presets[j.preset_index].features=='background')
        self.assertIn('background',job.features)
        params=dict(n_estimators=30,max_depth=2,learning_rate=.2,min_child_weight=1,reg_lambda=1)
        train=job.data.copy();train.y=train.background * 10
        model,enc=fit(train,job.features,job.categorical,params,self.cfg)
        test=train.iloc[:5].copy();first=model.predict(enc.transform(test));test.background+=5
        self.assertGreater(np.abs(first-model.predict(enc.transform(test))).max(),1)
        c=copy.deepcopy(self.cfg);c.presets=[ExperimentPreset(features='background',reference_ids=['S00'])]
        jobs,audit=prepare_project(c);self.assertFalse(jobs);self.assertTrue(all('study stations' in r['reason'] for r in audit['reports']))

    def test_balanced_differs_from_monthly(self):
        dates=pd.date_range('2025-01-01','2025-03-01',freq='D',inclusive='left')
        raw=pd.DataFrame(dict(station_id='A',date=dates,y=[1 if t.month==1 else 5 for t in dates]))
        v=VariableSpec(frequency='D')
        monthly,_=aggregate(raw,v,ExperimentPreset(grain='monthly'))
        balanced,_=aggregate(raw,v,ExperimentPreset(grain='balanced'))
        self.assertEqual(len(monthly),2);self.assertEqual(balanced.y.iloc[0],3)

    def test_single_station_association_unavailable(self):
        weather_path = self.cfg.variables[0].source.path
        weather = pd.read_csv(weather_path)
        later = weather.copy(); later['datetime'] = pd.to_datetime(later['datetime']) + pd.Timedelta(days=21)
        pd.concat([weather,later]).to_csv(weather_path,index=False)
        s=self.cfg.shapes[0].source
        frame=pd.read_csv(s.path).iloc[:1];frame.to_csv(s.path,index=False)
        jobs,audit=prepare_project(self.cfg)
        self.assertTrue(any('single station' in r['reason'] for r in audit['reports']))
        self.assertTrue(any(self.cfg.presets[j.preset_index].task=='forecast' for j in jobs))

    def test_end_to_end_export_reload_and_cancel(self):
        self.cfg.presets=self.cfg.presets[:1]
        before=Path(self.cfg.variables[0].source.path).read_bytes()
        out=run_project(self.cfg)
        self.assertEqual(Path(self.cfg.variables[0].source.path).read_bytes(),before)
        self.assertEqual(json.loads((out/'status.json').read_text())['status'],'completed')
        summary=pd.read_csv(out/'summary.csv');self.assertEqual(len(summary),2)
        folder=out/summary.experiment.iloc[0]
        frame=pd.read_csv(folder/'prepared.csv')
        model,encoder,_=load_bundle(folder/'final')
        np.testing.assert_allclose(model.predict(encoder.transform(frame)),predict_frame(folder/'final',frame))
        self.assertTrue((folder/'treeshap.svg').exists())
        self.assertTrue((folder/'fold_1'/'manifest.json').exists())
        stop=threading.Event();stop.set()
        with self.assertRaises(InterruptedError):run_project(self.cfg,cancel=stop)
        statuses=[json.loads(p.read_text(encoding='utf-8'))['status'] for p in Path(self.cfg.output_dir).glob('*/status.json')]
        self.assertIn('cancelled',statuses)

    def test_forecast_inference_next_period(self):
        self.cfg.presets=[self.cfg.presets[1]]
        out=run_project(self.cfg)
        summary=pd.read_csv(out/'summary.csv');folder=out/summary.experiment.iloc[0]/'final'
        pred=predict_project(folder,self.cfg)
        self.assertEqual(len(pred),8)
        self.assertTrue((pred.date==pd.Timestamp('2025-01-22')).all())

    def test_constant_score_undefined(self):
        self.assertIsNone(scores([1,1],[1,1])['r2'])

    def test_wide_aliases_and_single_station(self):
        path=self.root/'wide.csv'
        pd.DataFrame({'code':['local'], '2025-01-01':[3.], '2025-01-02':[4.]}).to_csv(path,index=False)
        v=VariableSpec(frequency='D',source=TableSource(path=str(path),station='code',layout='wide',id_map={'local':'canonical'}))
        data,_=load_variable(v)
        self.assertEqual(data.y.tolist(),[3.,4.]);self.assertEqual(set(data.station_id),{'canonical'})
        pd.DataFrame({'time':['2025-01-01'],'value':[2]}).to_csv(path,index=False)
        v=VariableSpec(column='value',frequency='D',source=TableSource(path=str(path),time='time',station='',single_station='A'))
        data,_=load_variable(v);self.assertEqual(data.station_id.iloc[0],'A')

    def test_forecast_background_uses_previous_period(self):
        c=copy.deepcopy(self.cfg);c.presets=[ExperimentPreset(grain='daily',task='forecast',features='background',reference_ids=['S08','S09'])]
        jobs,_=prepare_project(c);self.assertTrue(jobs)
        raw,_=load_variable(c.variables[0]);reference=raw[raw.station_id.isin(['S08','S09'])]
        ag,_=aggregate(reference,c.variables[0],c.presets[0]);means=ag.groupby('date').y.mean()
        job=jobs[0];expected=job.data.date.map(lambda t: means.loc[t-pd.Timedelta(days=1)])
        np.testing.assert_allclose(job.data.background,expected)

    def test_relative_project_paths(self):
        c=asdict(self.cfg);c['variables'][0]['source']['path']='weather.csv';c['output_dir']='runs'
        dump(c,self.root/'project.json')
        restored,_=load_project_file(self.root/'project.json')
        self.assertEqual(Path(restored.variables[0].source.path),self.root/'weather.csv')
        self.assertEqual(Path(restored.output_dir),self.root/'runs')

    def test_daynight_model_preserves_segment_and_unlabelled_inference(self):
        self.cfg.presets=[ExperimentPreset(name='昼夜',grain='daynight')]
        out=run_project(self.cfg);summary=pd.read_csv(out/'summary.csv')
        row=summary[(summary.target=='Air temperature') & summary['group'].str.contains('day')].iloc[0]
        folder=out/row.experiment/'final';_,_,manifest=load_bundle(folder)
        self.assertEqual(manifest['segment'],'day')
        path=self.cfg.variables[0].source.path
        weather=pd.read_csv(path).drop(columns=['temperature']);weather.to_csv(path,index=False)
        result=predict_project(folder,self.cfg)
        self.assertTrue((result.date.dt.hour==6).all())
        self.assertEqual(len(result),21*8)

    def test_separate_background_files_and_unit_guard(self):
        c=copy.deepcopy(self.cfg);c.presets=[ExperimentPreset(grain='daily',features='background')]
        weather=pd.read_csv(c.variables[0].source.path)
        bgpath=self.root/'background.csv';weather[weather.station_id.isin(['S08','S09'])].to_csv(bgpath,index=False)
        c.backgrounds=copy.deepcopy(c.variables)
        for v in c.backgrounds:v.source.path=str(bgpath)
        jobs,_=prepare_project(c);self.assertEqual(len(jobs),2)
        c.backgrounds[0].unit='K';jobs,audit=prepare_project(c)
        self.assertEqual(len(jobs),1);self.assertIn('units',audit['reports'][0]['reason'])

    def test_invalid_config_and_circular_target(self):
        for key,value in [('n_jobs',0),('outer_folds',1),('output_dir','')]:
            c=copy.deepcopy(self.cfg);setattr(c,key,value)
            with self.assertRaises(ValueError):validate(c)
        c=copy.deepcopy(self.cfg);c.variables[0].name='风向'
        with self.assertRaisesRegex(ValueError,'circular'):validate(c)

    def test_parameter_selected_from_inner_trials(self):
        self.cfg.grid.append(dict(n_estimators=6,max_depth=1,learning_rate=.2,min_child_weight=1,reg_lambda=2))
        jobs,_=prepare_project(self.cfg);j=jobs[0]
        selected,trials=tune(j.data,j,self.cfg,threading.Event())
        self.assertEqual(selected,min(trials,key=lambda t:t['mse'])['params'])

    def test_monthly_short_series_and_missing_month(self):
        dates=pd.to_datetime(['2025-01-01','2025-03-01'])
        d=pd.DataFrame(dict(station_id='A',date=dates,y=[1,3]))
        result,audit=aggregate(d,VariableSpec(frequency='MS'),ExperimentPreset(grain='balanced'))
        self.assertTrue(result.empty)
        with self.assertRaisesRegex(ValueError,'at least'):forecast_features(d,'monthly')


if __name__=='__main__':unittest.main()
