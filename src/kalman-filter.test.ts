import { KalmanFilter } from "./kalman-filter";
import fs from "fs";

import testData from "../fixtures/test-data.json";
import testData2 from "../fixtures/test-data-2.json";

describe("KalmanFilter", function () {
  let kf: KalmanFilter;

  beforeEach(function () {
    kf = new KalmanFilter(0.2, 0.0015);
  });

  it("should initialize the Kalman filter", function () {
    kf.init(0);
    expect(kf.kf).not.toBeUndefined();
    expect(kf.state).not.toBeNull();
    expect(kf.previousCorrected).not.toBeNull();
    expect(typeof kf.lastTS).toBe("number");
  });

  it("should predict the next state", function () {
    kf.init(0);
    const steptime = kf.predict();
    expect(kf.state).not.toBeNull();
  });

  it("should correct the state with a new observation", function () {
    kf.init(0);
    kf.predict();
    kf.correct(1, 1);
    expect(kf.state).not.toBeNull();
  });

  it("should return the mean of the state", function () {
    kf.init(0);
    kf.predict();
    kf.correct(1, 1);
    const mean = kf.mean();
    expect(Array.isArray(mean)).toBe(true);
    expect(mean).toHaveLength(2);
  });

  it("should return the count of the state updates", function () {
    kf.init(0);
    kf.predict();
    kf.correct(1, 1);
    const count = kf.count();
    expect(typeof count).toBe("number");
  });

  it("should use test data", function () {
    const startData = testData[0];
    const initTs = new Date(startData.created_at).getTime();
    // kf = new KalmanFilter(0.2, 0.0015);
    kf.init(startData.field5, initTs);
    const out = [
      { ts: initTs, in: startData.field5, out: kf.mean()[0], state: kf.state },
    ];
    testData.slice(1).forEach((data, index) => {
      const ts = new Date(data.created_at).getTime();
      kf.predict(ts);
      kf.correct(data.field5, ts);
      const mean = kf.mean();
      out.push({ ts, in: data.field5, out: mean[0], state: kf.state });
    });
    fs.writeFileSync(
      `./fixtures/kalman-filter-out.json`,
      JSON.stringify(out, null, 2)
    );
  });

  it("should use test data with extra predictions", function () {
    const startData = testData[0];
    const initTs = new Date(startData.created_at).getTime();
    kf.init(startData.field5, initTs);
    const out = [
      {
        ts: initTs,
        in: startData.field5 as number | undefined,
        out: kf.mean()[0],
        state: kf.state,
      },
    ];
    testData.slice(1).forEach((data, index) => {
      const ts = new Date(data.created_at).getTime();
      kf.predict(ts);
      kf.correct(data.field5, ts);
      const mean = kf.mean();
      out.push({ ts, in: data.field5, out: mean[0], state: kf.state });
      kf.predict(ts + 5 * 60e3);
      out.push({
        ts: ts + 5 * 60e3,
        in: undefined,
        out: kf.mean()[0],
        state: kf.state,
      });
      kf.predict(ts + 10 * 60e3);
      out.push({
        ts: ts + 10 * 60e3,
        in: undefined,
        out: kf.mean()[0],
        state: kf.state,
      });
    });
    fs.writeFileSync(
      `./fixtures/kalman-filter-inter-out.json`,
      JSON.stringify(out, null, 2)
    );
  });

  it("should use test data 2", function () {
    const startData = testData2[0];
    const initTs = new Date(startData.created_at).getTime();
    // kf = new KalmanFilter(0.2, 0.0015);
    kf.init(startData.field5, initTs);
    const out = [
      {
        ts: initTs,
        in: startData.field5,
        out: kf.mean()[0],
        state: kf.state,
        nis: kf.nis,
        logL: kf.logL,
      },
    ];
    testData2.slice(1).forEach((data, index) => {
      const ts = new Date(data.created_at).getTime();
      kf.predict(ts);
      kf.correct(data.field5, ts);
      const mean = kf.mean();
      out.push({
        ts,
        in: data.field5,
        out: mean[0],
        state: kf.state,
        nis: kf.nis,
        logL: kf.logL,
      });
    });
    fs.writeFileSync(
      `./fixtures/kalman-filter-data2-out.json`,
      JSON.stringify(out, null, 2)
    );
    console.log(
      "The output results have been checked by o4-mini (feed it two samples)"
    );
  });
});

/*

TODO:

- IMM - Interacting Multiple Model
- Keep two model probabilities mu1 mu2
- Markov transition model Probabilities [[0.95, 0.2], [0.05, 0.8]] ?? to c[i] / mu_bar[i]
- Calculate omega matrix aka mixing weights and 
    // 1) Compute next‐step mixing weights
    let muBar = [
      T[0][0]*mu[0] + T[0][1]*mu[1],
      T[1][0]*mu[0] + T[1][1]*mu[1]
    ];
    let omega = [
      [T[0][0]*mu[0]/muBar[0],  T[0][1]*mu[1]/muBar[0]],
      [T[1][0]*mu[0]/muBar[1],  T[1][1]*mu[1]/muBar[1]]
    ];
- Update state and covariance for both filters (X = omega[0][i] * xi, P = omega[0][i] * (Pi + (xi - ximix) * (xi - ximix)'), ...)
- Predict and correct with likelihoods
- Update model probabilities with Bayes
    (using mu_bar[i] aka c[i], not mu[i])
    let norm = lik[0]*mu_bar[0] + lik[1]*mu_bar[1];
    mu = [ (lik[0]*mu_bar[0]/norm), (lik[1]*mu_bar[1]/norm) ];
- Output fused estimate

*/
