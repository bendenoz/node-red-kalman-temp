import { KalmanFilter } from './kalman-filter';
import fs from 'fs';

import testData from '../fixtures/test-data.json';

describe('KalmanFilter', function () {
  let kf: KalmanFilter;

  beforeEach(function () {
    kf = new KalmanFilter(0.2, 0.0015);
  });

  it('should initialize the Kalman filter', function () {
    kf.init(0);
    expect(kf.kf).not.toBeUndefined();
    expect(kf.state).not.toBeNull();
    expect(kf.previousCorrected).not.toBeNull();
    expect(typeof kf.lastTS).toBe('number');
  });

  it('should predict the next state', function () {
    kf.init(0);
    const steptime = kf.predict();
    expect(kf.state).not.toBeNull();
  });

  it('should correct the state with a new observation', function () {
    kf.init(0);
    kf.predict();
    kf.correct(1, 1);
    expect(kf.state).not.toBeNull();
  });

  it('should return the mean of the state', function () {
    kf.init(0);
    kf.predict();
    kf.correct(1, 1);
    const mean = kf.mean();
    expect(Array.isArray(mean)).toBe(true);
    expect(mean).toHaveLength(2);
  });

  it('should return the count of the state updates', function () {
    kf.init(0);
    kf.predict();
    kf.correct(1, 1);
    const count = kf.count();
    expect(typeof count).toBe('number');
  });

  it('should use test data', function () {
    const startData = testData[0];
    const initTs = new Date(startData.created_at).getTime()
    // kf = new KalmanFilter(0.2, 0.0015);
    kf.init(startData.field5, initTs);
    const out = [{ ts: initTs, in: startData.field5, out: kf.mean()[0], state: kf.state }];
    testData.slice(1).forEach((data, index) => {
      const ts = new Date(data.created_at).getTime();
      kf.predict(ts);
      kf.correct(data.field5, ts);
      const mean = kf.mean();
      out.push({ ts, in: data.field5, out: mean[0], state: kf.state });
    });
    fs.writeFileSync(`./fixtures/kalman-filter-out.json`, JSON.stringify(out, null, 2));
  })

  it('should use test data with extra predictions', function () {
    const startData = testData[0];
    const initTs = new Date(startData.created_at).getTime()
    kf.init(startData.field5, initTs);
    const out = [{ ts: initTs, in: startData.field5 as number | undefined, out: kf.mean()[0], state: kf.state }];
    testData.slice(1).forEach((data, index) => {
      const ts = new Date(data.created_at).getTime();
      kf.predict(ts);
      kf.correct(data.field5, ts);
      const mean = kf.mean();
      out.push({ ts, in: data.field5, out: mean[0], state: kf.state });
      kf.predict(ts + 5 * 60e3);
      out.push({ ts: ts + 5 * 60e3, in: undefined, out: kf.mean()[0], state: kf.state });
      kf.predict(ts + 10 * 60e3);
      out.push({ ts: ts + 10 * 60e3, in: undefined, out: kf.mean()[0], state: kf.state });
    });
    fs.writeFileSync(`./fixtures/kalman-filter-inter-out.json`, JSON.stringify(out, null, 2));
  })
});