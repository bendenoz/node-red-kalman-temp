import { Node, NodeDef, NodeAPI } from "node-red";
import { KalmanFilter } from "./kalman-filter";
import { performance } from "perf_hooks";

export interface KalmanTempNodeDef extends NodeDef {
  R?: number;
  Q?: number;
  predictInterval?: number;
  lookAhead?: number;
  fastQFactor?: number;
  topic?: string;
}

// Markov / fusion parameters (allocated once)
const P11 = 0.995;
const P22 = 0.95;
const Pi = [
  [P11, 1 - P11],
  [1 - P22, P22],
];

interface Props {
  kfSlow: KalmanFilter;
  kfFast: KalmanFilter;
  // dynamic fusion state
  mu: [number, number];
}

module.exports = function (RED: NodeAPI) {
  function KalmanTempNode(this: Node, config: KalmanTempNodeDef) {
    RED.nodes.createNode(this, config);
    const node = this;

    // Set default values using nullish coalescing operator
    const R = config.R ?? 0.2;
    const Q = config.Q ?? 0.001;
    const interval = Math.max(config.predictInterval ?? 60, 1) * 1000; // Ensure interval is positive
    const lookAhead = Math.max(config.lookAhead ?? 0, 0) * 1000; // Ensure lookAhead is non-negative
    // fastQFactor may be missing on node upgrade; default to 1
    let fastQFactor = Number((config as any).fastQFactor ?? 1);
    if (!isFinite(fastQFactor) || fastQFactor <= 0) {
      fastQFactor = 1;
    }

    let props: Props | undefined;
    let timeoutId: NodeJS.Timeout | undefined;

    const initProps = (initValue: number, initTs: number) => {
      const slow = new KalmanFilter(R, Q);
      const fast = new KalmanFilter(R, Q * fastQFactor);
      slow.init(initValue, initTs);
      fast.init(initValue, initTs);
      props = {
        kfSlow: slow,
        kfFast: fast,
        mu: [0, 1],
      };
    };

    const schedulePrediction = (sendCallback: (now: number) => void) => {
      clearTimeout(timeoutId);
      timeoutId = setTimeout(() => {
        const now = performance.now();
        sendCallback(now);
        schedulePrediction(sendCallback);
      }, interval);
    };

    node.on("input", (msg, send, done) => {
      const pv = Number(msg.payload);
      if (isNaN(pv) || !isFinite(pv)) {
        return node.warn("Input must be a valid number.");
      }

      // Node-RED 0.x compat - https://nodered.org/docs/creating-nodes/node-js#sending-messages
      const nodeSend =
        send ||
        function (...args) {
          node.send.apply(node, args);
        };

      const now = performance.now();
      if (!props) {
        initProps(pv, now);
      }

      // update filters with new measurement
      props!.kfSlow.predict(now);
      props!.kfSlow.correct(pv, now);

      props!.kfFast.predict(now);
      props!.kfFast.correct(pv, now);

      // Multiple-Model Adaptive Estimation (MMAE):
      // - compute per-model log-likelihoods (from each Kalman filter)
      // - apply Markov-model prior (Pi) to get muPrior
      // - weight by likelihoods (numerically stable via maxLogL) and normalize
      // The resulting `props.mu` are the model posterior probabilities (confidence)
      // for [slow, fast] models and are used to fuse predictions below.
      // Markov chain weights update (use shared constants)
      const logL = [
        props!.kfSlow.logL ?? Number.NEGATIVE_INFINITY,
        props!.kfFast.logL ?? Number.NEGATIVE_INFINITY,
      ];
      const muPrior = [
        Pi[0][0] * props!.mu[0] + Pi[1][0] * props!.mu[1],
        Pi[0][1] * props!.mu[0] + Pi[1][1] * props!.mu[1],
      ];
      const maxLogL = Math.max(...logL);
      let w = muPrior.map((m, i) => m * Math.exp(logL[i] - maxLogL));
      let muSum = w.reduce((a, b) => a + b, 0);
      if (!isFinite(muSum) || muSum === 0) {
        w = [0.5, 0.5];
        muSum = 1;
      }
      props!.mu = [w[0] / muSum, w[1] / muSum];

      const sendValue = (sendNow: number) => {
        const predictionTime = sendNow + lookAhead;
        // make prediction for look-ahead using fused strategy: predict both and fuse
        props!.kfSlow.predict(predictionTime);
        props!.kfFast.predict(predictionTime);
        const slowP = props!.kfSlow.mean()[0] ?? 0;
        const fastP = props!.kfFast.mean()[0] ?? 0;
        const fusedP = props!.mu[0] * slowP + props!.mu[1] * fastP;

        nodeSend([
          {
            ...(msg as any),
            payload: fusedP,
            topic: config.topic || msg.topic,
            // expose slow-model confidence (mu[0]) as a lightweight metric
            slowConfidence: props!.mu[0],
          } as any,
        ] as any);
      };
      sendValue(now);
      schedulePrediction(sendValue);
    });

    node.on("close", () => {
      clearTimeout(timeoutId);
      props = undefined;
    });
  }

  RED.nodes.registerType("kalman-temp", KalmanTempNode);
};
