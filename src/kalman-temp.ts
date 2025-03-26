import { Node, NodeDef, NodeAPI } from "node-red";
import { KalmanFilter } from "./kalman-filter";
import { performance } from "perf_hooks";

export interface KalmanTempNodeDef extends NodeDef {
  R?: number;
  Q?: number;
  predictInterval?: number;
  lookAhead?: number;
  topic?: string;
}

interface Props {
  kf: KalmanFilter;
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

    let props: Props | undefined;
    let timeoutId: NodeJS.Timeout | undefined;

    const initProps = (initValue: number, initTs: number) => {
      props = { kf: new KalmanFilter(R, Q) };
      props.kf.init(initValue, initTs);
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
      } else {
        const st = props.kf.predict(now);
        props.kf.correct(pv, now);
      }

      const sendValue = (sendNow: number) => {
        const predictionTime = sendNow + lookAhead;
        props!.kf.predict(predictionTime);
        const [value] = props!.kf.mean();
        nodeSend([
          {
            ...msg,
            payload: value,
            topic: config.topic || msg.topic,
          },
        ]);
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
