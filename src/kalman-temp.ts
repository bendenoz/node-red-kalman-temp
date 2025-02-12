import { Node, NodeDef, NodeAPI } from "node-red";
import { KalmanFilter } from "./kalman-filter";

export interface KalmanTempConfig {
    R: number;
    Q: number;
    predictInterval: number;
}
export interface KalmanTempNodeDef extends NodeDef, KalmanTempConfig { }

interface Props {
    kf: KalmanFilter;
}

module.exports = function (RED: NodeAPI) {
    function KalmanTempNode(this: Node, config: KalmanTempNodeDef) {
        RED.nodes.createNode(this, config);
        const node = this;

        // Default values: R = 0.2°C, Q = 0.0015 (deg/min²)
        const R = config.R ?? 0.2;
        const Q = config.Q ?? 0.001;
        const interval = config.predictInterval ?? 60;

        let props: Props | undefined;
        let timeoutId: NodeJS.Timeout | undefined;

        const initProps = (initValue: number, initTs: number) => {
            props = {
                kf: new KalmanFilter(R, Q)
            }
            props.kf.init(initValue, initTs);
        }

        const predict = () => {
            const now = performance.now();
            props!.kf.predict(now);
            const [value] = props!.kf.mean();
            node.send([{ payload: value }]);
        }

        const mainLoop = () => {
            if (timeoutId) {
                clearTimeout(timeoutId);
            }
            timeoutId = setTimeout(() => {
                predict();
                mainLoop();
            }, interval * 1000);
        }

        node.on("input", (msg) => {
            const pv = Number(msg.payload);
            const now = performance.now();

            if (pv !== null && !isNaN(pv) && isFinite(pv)) {
                if (props) {
                    const st = props.kf.predict(now);
                    props.kf.correct(pv, st);
                } else {
                    initProps(pv, now);
                }
                const [value] = props!.kf.mean();
                node.send([{ payload: value }]);
                mainLoop();
            } else {
                node.warn("Input must be a number.");
            }
        });

        node.on("close", () => {
            if (timeoutId) {
                clearTimeout(timeoutId);
            }
            props = undefined;
        });
    }

    RED.nodes.registerType("kalman-temp", KalmanTempNode);
};
