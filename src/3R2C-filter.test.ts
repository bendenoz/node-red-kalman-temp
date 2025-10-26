import KalmanClass from "kalman-filter/lib/kalman-filter";
import StateType from "kalman-filter/lib/state";
import fs from "fs";
import path from "path";

/*
--- Identified Parameters ---
R1 (Internal-Mass):   0.0016 K/W
R2 (Internal-Outside): 0.2630 K/W
R3 (Mass-Outside):    0.0080 K/W
Ci (Internal Cap.):   3.140 MJ/K
Cm (Mass Cap.):       16.681 MJ/K
Tm_initial (Mass T):  11.41 °C
Q_bias (Internal Gains): 1080.54 W
*/

const R1 = 0.0016;
const R2 = 0.263; // ignored for now
const R3 = 0.008;
const Ci = 3.14e6;
const Cm = 10.681e6;

describe("3R2C filter (stub)", () => {
  test("upstream kalman-filter package import exists", () => {
    expect(KalmanClass).toBeDefined();
  });

  test("load fixture feeds_202502_regression.csv", () => {
    const csvPath = path.resolve(
      __dirname,
      "../fixtures/feeds_202502_regression.csv"
    );
    const raw = fs.readFileSync(csvPath, "utf8");
    expect(raw).toBeTruthy();
    const lines = raw
      .split(/\r?\n/)
      .map((l) => l.trim())
      .filter(Boolean)
      .slice(1); // skip header
    // parse CSV rows into arrays of columns
    const rows = lines.map((l) => l.split(",").map((c) => c.trim()));
    // Expect at least a header + one data row
    expect(lines.length).toBeGreaterThanOrEqual(2);

    const stDvTC = 0.8;

    const [initTime, initTin, initTout, , initQheat] = rows[0];
    let currentTS = new Date(initTime).getTime();

    // see also this discussion https://gemini.google.com/app/39d791991948de4f
    const kf = new KalmanClass({
      observation: {
        dimension: 1, // Tin
        // R
        covariance: [[stDvTC ** 2]],
      },
      dynamic: {
        dimension: 3, // Tin, Tmass, Qbias
        init: {
          mean: [[Number(initTin)], [Number(initTin)], [800]],
          // Initial P
          covariance: [10 ** 2, 10 ** 2, 1000 ** 2],
          index: -1,
        },
        // F (or A)
        transition: ({ steptime }: { steptime: number }) => {
          const dT = steptime; // in seconds
          return [
            [1 - dT / (R1 * Ci), dT / (R1 * Ci), dT / Ci],
            [dT / (R1 * Cm), 1 - dT / (R1 * Cm) - dT / (R3 * Cm), 0],
            [0, 0, 1],
          ];
        },
        // G
        constant: ({
          steptime,
          u,
        }: {
          steptime: number;
          u: { tOut: number; QHeat: number };
        }) => {
          const dt = steptime; // in seconds
          const { tOut, QHeat } = u; // in °C and W
          // This constant term represents external inputs to the system.
          // It is added to the state vector at each step.
          // For Tin: effect of Qheat (internal gains) on internal temperature.
          // For Tmass: effect of Tout (outside temp) on mass temperature.
          // For Qbias: no direct external input.
          // Note: the Tmass dependence on the previous state (i.e. the -Tmass
          // term) is represented in the transition matrix above
          // (the 1 - dT/(R1*Cm) - dT/(R3*Cm) entry). When using forward
          // Euler discretization the state-dependent Tmass term belongs in
          // the transition matrix while the pure external drive from Tout
          // appears in this constant vector as dt * Tout / (R3 * Cm).
          // Including a -(dt * Tmass) term here would double-count the
          // Tmass contribution.
          return [[(dt * QHeat) / Ci], [(dt * tOut) / (R3 * Cm)], [0]];
        },
        // Q - steptime in seconds
        covariance: ({ steptime }: { steptime: number }) => {
          const dtMin = steptime / 60;
          const qTin = 0.08 ** 2 * dtMin;
          const qTmass = 0.08 ** 2 * dtMin;
          const qQbias = 0.005 ** 2 * dtMin;
          return [
            [qTin, 0, 0],
            [0, qTmass, 0],
            [0, 0, qQbias],
          ];
        },
      },
    });

    let previousCorrected: StateType | null = null;

    const result: Number[][] = [];

    rows.slice(1).forEach((cols) => {
      const [time, tIn, tOut, , QHeat] = cols;
      const ts = new Date(time).getTime();
      const steptime = (ts - currentTS) / 1000; // in seconds
      currentTS = ts;
      const predicted = kf.predict({
        previousCorrected,
        steptime,
        u: { tOut: Number(tOut), QHeat: Number(QHeat) },
      });
      previousCorrected = kf.correct({
        predicted,
        observation: [Number(tIn)],
        steptime,
      });
      if (previousCorrected) {
        const {
          mean: [[xTIn], [xTmass], [xQbias]],
          index,
        } = previousCorrected;
        result.push([ts, tIn, xTIn, xTmass, xQbias / 100, tOut]);
      }
    });
    console.log(result);
    const tsvLines = result.map((row) => row.join("\t")).join("\n");
    const outPath = path.resolve(
      __dirname,
      "../fixtures/feeds_202502_regression_result.tsv"
    );
    fs.writeFileSync(outPath, tsvLines, "utf8");
  });
});
