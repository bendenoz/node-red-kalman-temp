import { Node, NodeDef, NodeAPI } from "node-red";
import { KalmanFilter } from "./kalman-filter";
import { performance } from 'perf_hooks';

export interface KalmanTempNodeDef extends NodeDef {
    R?: number;
    Q?: number;
    predictInterval?: number;
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

        let props: Props | undefined;
        let timeoutId: NodeJS.Timeout | undefined;

        const initProps = (initValue: number, initTs: number) => {
            props = { kf: new KalmanFilter(R, Q) };
            props.kf.init(initValue, initTs);
        };

        const predict = () => {
            if (!props) return;
            const now = performance.now();
            props.kf.predict(now);
        };

        const schedulePrediction = (sendCallback: () => void) => {
            clearTimeout(timeoutId);
            timeoutId = setTimeout(() => {
                predict();
                sendCallback();
                schedulePrediction(sendCallback);
            }, interval);
        };

        node.on("input", (msg) => {
            const pv = Number(msg.payload);
            if (isNaN(pv) || !isFinite(pv)) {
                return node.warn("Input must be a valid number.");
            }

            const now = performance.now();
            if (!props) {
                initProps(pv, now);
            } else {
                const st = props.kf.predict(now);
                props.kf.correct(pv, st);
            }

            const sendValue = () => {
                const [value] = props!.kf.mean();
                node.send([{ ...msg, payload: value }]);
            }
            sendValue();
            schedulePrediction(sendValue);
        });

        node.on("close", () => {
            clearTimeout(timeoutId);
            props = undefined;
        });
    }

    RED.nodes.registerType("kalman-temp", KalmanTempNode);
};
