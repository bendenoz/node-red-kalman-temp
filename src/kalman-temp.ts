import { Node, NodeDef, NodeAPI } from "node-red";
import { KalmanFilter } from "./kalman-filter";

interface KalmanTempNodeDef extends NodeDef {
    R: number;
    Q: number;
}

interface Props {
    kf: KalmanFilter;
}

module.exports = function (RED: NodeAPI) {
    function KalmanTempNode(this: Node, config: KalmanTempNodeDef) {
        RED.nodes.createNode(this, config);
        const node = this;

        // Default values: R = 0.2°C, Q = 0.0015 (deg/min²)
        const R = config.R ?? 0.2;
        const Q = config.Q ?? 0.0015;

        let props: Props | undefined;

        const initProps = (initValue: number, initTs: number) => {
            props = {
                kf: new KalmanFilter(R, Q)
            }
            props.kf.init(initValue, initTs);
        }

        node.on("input", (msg) => {
            const pv = Number(msg.payload);
            const now = performance.now();

            if (pv !== null && !isNaN(pv) && isFinite(pv)) {
                if (props) {
                    const ts = props.kf.predict(now);
                    props.kf.correct(pv, now);
                    const [value] = props.kf.mean();
                    node.send([{ payload: value }]);
                }
                else {
                    initProps(pv, now);
                }
            } else {
                node.warn("Input must be a number.");
            }
        });

        node.on("close", () => {
            props = undefined;
        });
    }

    RED.nodes.registerType("kalman-temp", KalmanTempNode);
};
