import KalmanClass from "kalman-filter/lib/kalman-filter";
import { add, subtract as sub, matMul, invert, transpose } from "simple-linalg";
import arrayToMatrix from "kalman-filter/lib/utils/array-to-matrix";
import StateType from "kalman-filter/lib/state";
import { performance } from "perf_hooks";

require("kalman-filter"); // must be required to init default models

/** Unused - for reference only */
class KalmanWithNIS extends KalmanClass {
  correct(options: any): StateType & { nis: number; logLikelihood: number } {
    const { predicted, observation } = options;

    const coreObservation = arrayToMatrix({
      observation,
      dimension: this.observation.dimension,
    } as any);
    const getValueOptions = Object.assign({}, options, {
      observation: coreObservation,
      index: predicted.index,
    });
    /** aka H */
    const stateProjection = this.getValue(
      this.observation.stateProjection,
      getValueOptions
    );

    const innovation = sub(
      coreObservation,
      this.getPredictedObservation({ stateProjection, opts: getValueOptions })
    );

    // calculate Normalized Innovation Squared (NIS)

    const R = this.getValue(this.observation.covariance, getValueOptions);

    /** S = H · P · Hᵀ + R */
    const S = add(
      matMul(
        matMul(stateProjection, predicted.covariance),
        transpose(stateProjection)
      ),
      R
    );

    const S_inv = invert(S);

    /** NIS = innovationᵀ·inv(S)·innovation -- aka squared [Mahalanobis](https://en.wikipedia.org/wiki/Mahalanobis_distance)? */
    const nis = matMul(matMul(transpose(innovation), S_inv), innovation)[0][0];

    // Calculate multivariate normal PDF likelihood
    const n = this.observation.dimension; // 1
    let detS: number;
    if (S.length === 1) {
      // actually S.length === 1 in our case
      detS = S[0][0];
    } else if (S.length === 2) {
      detS = S[0][0] * S[1][1] - S[0][1] * S[1][0];
    } else {
      // Extend determinant calculation as needed for higher dimensions
      throw new Error(
        "Determinant calculation for matrices >2x2 not implemented."
      );
    }

    // Compute necessary logarithms
    const logDetS = Math.log(detS); // log determinant of S
    const log2Pi = Math.log(2 * Math.PI);

    // Compute the log-likelihood using the multivariate normal PDF formula
    // logLikelihood = -0.5 * (nis + n * log(2π) + log(detS))
    const logLikelihood = -0.5 * (nis + n * log2Pi + logDetS);

    const result = super.correct(options);
    (result as any).nis = nis;
    (result as any).logLikelihood = logLikelihood;
    return result as StateType & { nis: number; logLikelihood: number };
  }
}

export class KalmanFilter {
  kf: KalmanClass | undefined;

  /** last predicted or corrected state */
  state: StateType | null = null;
  /** last corrected state */
  previousCorrected: StateType | null = null;

  /**
   * previous timestamp, in milliseconds
   */
  lastTS: number;

  /** R */
  R: number;

  /** Q - in degrees per minute per squared minute */
  Q: number;

  /** log-likelihood */
  logL: number;
  /** Normalized Innovation Squared */
  nis: number;

  constructor(R: number, Q: number) {
    this.lastTS = -1;
    this.R = R;
    this.Q = Q;
    this.logL = Number.NEGATIVE_INFINITY;
    this.nis = Number.POSITIVE_INFINITY;
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
          covariance: ({ steptime }: { steptime: number }) => {
            const dTmin = steptime / 60;
            // Continuous White Acceleration Noise (CWAN) model
            const qNoise = this.Q ** 2;
            const rateVar = qNoise * dTmin;
            const valueVar = (qNoise * dTmin ** 3) / 3;
            const covar = (qNoise * dTmin ** 2) / 2;
            return [
              [valueVar, covar],
              [covar, rateVar],
            ];
          },
        },
      });
    } catch (e) {
      console.error(e);
      throw e;
    }
    this.state = this.kf.getInitState();
    this.previousCorrected = this.state;
    this.lastTS = initTs;
  }

  predict(ts = performance.now()) {
    if (!this.kf || !this.lastTS) throw new Error("No KF instance");
    const steptime = (ts - this.lastTS) / 1000;
    const predicted = this.kf.predict({
      previousCorrected: this.previousCorrected,
      steptime,
    });
    this.state = predicted;
  }

  /**
   * @param value
   * @param ts
   */
  correct(value: number, ts: number) {
    if (!this.kf) throw new Error("No KF instance");
    if (!this.state) throw new Error("No State");

    /** aka H ([[1,0]] here) */
    const stateProjection = this.kf.getValue(
      this.kf.observation.stateProjection
    );
    const predictedObservation = this.kf.getPredictedObservation({
      stateProjection,
      opts: { predicted: this.state },
    });

    /** ν (nu) */
    const innovation = sub([[value]], predictedObservation);

    // calculate Normalized Innovation Squared (NIS)

    const R = this.kf.observation.covariance; // call this.kf.getValue() if needed

    /** S = H · P · Hᵀ + R — could be simplified to P + R  for 1D case */
    const S = add(
      matMul(
        matMul(stateProjection, this.state.covariance),
        transpose(stateProjection)
      ),
      R
    );

    const S_inv = invert(S);

    /**
     * NIS = innovationᵀ·inv(S)·innovation — aka squared [Mahalanobis](https://en.wikipedia.org/wiki/Mahalanobis_distance)
     * Could be simplified to ν² / S for 1D case
     */
    const nis = matMul(matMul(transpose(innovation), S_inv), innovation)[0][0];

    const detS = S[0][0]; // 1x1 matrix !
    // Compute necessary logarithms
    const logDetS = Math.log(detS); // log determinant of S
    const log2Pi = Math.log(2 * Math.PI);

    // Compute the log-likelihood using the multivariate normal PDF formula
    // logLikelihood = -0.5 * (nis + n * log(2π) + log(detS))
    const logLikelihood = -0.5 * (nis + 1 * log2Pi + logDetS);

    this.logL = logLikelihood;
    this.nis = nis;

    const corrected = this.kf.correct({
      predicted: this.state,
      observation: [value],
    });
    this.previousCorrected = corrected;
    this.state = corrected;
    this.lastTS = ts;
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
