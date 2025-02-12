import KalmanClass from "kalman-filter/lib/kalman-filter";
import StateType from "kalman-filter/lib/state";

const { performance } = require("perf_hooks"); // not needed for node > ??

require("kalman-filter"); // must be required to init default models

export class KalmanFilter {
  kf: KalmanClass | undefined;

  state: StateType | null = null;

  /**
   * previous timestamp, in milliseconds
   */
  lastTS: number;

  /** R */
  R: number;

  /** Q - in degrees per minute per squared minute */
  Q: number

  constructor(R: number, Q: number) {
    this.lastTS = -1;
    this.R = R;
    this.Q = Q;
  }

  /**
   * Instanciate kalman filter with value
   */
  init(initValue: number, initTs = performance.now()) {
    try {
      this.kf = new KalmanClass({
        observation: {
          dimension: 1,
          // R
          covariance: [[this.R ** 2]],
        },
        dynamic: {
          dimension: 2,
          init: {
            mean: [[initValue], [0]],
            // Initial P
            covariance: [10 ** 2, 10 ** 2],
            index: -1,
          },
          transition: ({ steptime }: { steptime: number }) => {
            return [
              [1, steptime / 60],
              [0, 1],
            ];
          },
          /** Q (noise) */
          covariance: ({
            steptime,
          }: { steptime: number }) => {
            const dTmin = steptime / 60;
            const rateNoise = this.Q * dTmin ** .5;
            const valueNoise = rateNoise * dTmin;
            const correl = 1; // assume total correlation since one determines the other
            return [
              [valueNoise ** 2, correl * valueNoise * rateNoise],
              [correl * valueNoise * rateNoise, rateNoise ** 2],
            ];
          },
        },
      });
    } catch (e) {
      console.error(e);
      throw e;
    }
    this.state = this.kf.getInitState();
    this.lastTS = initTs;
  }

  /** Returns step duration in seconds, used by correct */
  predict(ts = performance.now()) {
    if (!this.kf || !this.lastTS) throw new Error("No KF instance");
    const steptime = (ts - this.lastTS) / 1000;
    const predicted = this.kf.predict({
      previousCorrected: this.state,
      steptime,
    });
    this.state = predicted;
    this.lastTS = ts;
    return steptime;
  }

  /**
   * @param value
   * @param steptime - maybe not needed?
   */
  correct(value: number, steptime: number) {
    if (!this.kf) throw new Error("No KF instance");
    const corrected = this.kf.correct({
      predicted: this.state,
      observation: [value],
      steptime,
    });

    this.state = corrected;
  }

  mean(): [number | null, number | null] {
    return this.state === null
      ? [null, null]
      : this.state.mean.map(([v]: [number]) => v);
  }

  count() {
    return this.state === null ? 0 : this.state.index;
  }
}