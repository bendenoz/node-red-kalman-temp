import { KalmanFilter } from "./kalman-filter";
import fs from "fs";
import testData2 from "../fixtures/test-data-2.json";

describe("KalmanFilter MAE comparison", () => {
  let kfSlow: KalmanFilter;
  let kfFast: KalmanFilter;

  beforeEach(() => {
    // Slow and fast filter parameters
    kfSlow = new KalmanFilter(0.8, 0.0015);
    kfFast = new KalmanFilter(0.8, 0.0015 * 10);
  });

  it("should process testData2 and compare MAE of slow and fast filters", () => {
    // ...existing code...
    const startData = testData2[0];
    const initTs = new Date(startData.created_at).getTime();
    kfSlow.init(startData.field5, initTs);
    kfFast.init(startData.field5, initTs);

    const actuals: number[] = [];
    const slowPreds: number[] = [];
    const fastPreds: number[] = [];
    const fusedPreds: number[] = [];
    const weights: number[][] = [];
    let mu = [0, 1]; // fast at first
    const p11 = 0.995;
    const p22 = 0.95;
    const Pi = [
      [p11, 1 - p11], // from slow -> [slow, fast]
      [1 - p22, p22], // from fast -> [slow, fast]
    ];

    testData2.slice(1).forEach((data) => {
      const ts = new Date(data.created_at).getTime();
      actuals.push(data.field5);

      kfSlow.predict(ts);
      kfSlow.correct(data.field5, ts);
      slowPreds.push(kfSlow.mean()[0] ?? 0);

      kfFast.predict(ts);
      kfFast.correct(data.field5, ts);
      fastPreds.push(kfFast.mean()[0] ?? 0);

      // Markov chain weight update
      const logL = [
        kfSlow.logL ?? Number.NEGATIVE_INFINITY,
        kfFast.logL ?? Number.NEGATIVE_INFINITY,
      ];
      const muPrior = [
        Pi[0][0] * mu[0] + Pi[1][0] * mu[1],
        Pi[0][1] * mu[0] + Pi[1][1] * mu[1],
      ];
      const maxLogL = Math.max(...logL);
      let w = muPrior.map((m, i) => m * Math.exp(logL[i] - maxLogL));
      let muSum = w.reduce((a, b) => a + b, 0);
      if (!isFinite(muSum) || muSum === 0) {
        w = [0.5, 0.5];
        muSum = 1;
      }
      mu = w.map((wi) => wi / muSum);
      weights.push([...mu]);

  const [meanSlowRaw] = kfSlow.mean() || [0, 0];
  const [meanFastRaw] = kfFast.mean() || [0, 0];
  const meanSlow = meanSlowRaw ?? 0;
  const meanFast = meanFastRaw ?? 0;
  fusedPreds.push(mu[0] * meanSlow + mu[1] * meanFast);
    });

    // Calculate MAE
    function mae(preds: number[], actuals: number[]) {
      return (
        preds.reduce((sum, pred, i) => sum + Math.abs(pred - actuals[i]), 0) /
        actuals.length
      );
    }

    const slowMAE = mae(slowPreds, actuals);
    const fastMAE = mae(fastPreds, actuals);
    const fusedMAE = mae(fusedPreds, actuals);

    console.log(`Slow filter MAE: ${slowMAE}`);
    console.log(`Fast filter MAE: ${fastMAE}`);
    console.log(`Fused filter MAE: ${fusedMAE}`);

    expect(typeof slowMAE).toBe("number");
    expect(typeof fastMAE).toBe("number");
    expect(typeof fusedMAE).toBe("number");
    expect(slowMAE).toBeGreaterThanOrEqual(0);
    expect(fastMAE).toBeGreaterThanOrEqual(0);
    expect(fusedMAE).toBeGreaterThanOrEqual(0);

    // Optionally write results to file
    fs.writeFileSync(
      "./fixtures/kalman-filter-mae-comparison.json",
      JSON.stringify(
        {
          slowMAE,
          fastMAE,
          fusedMAE,
          slowPreds,
          fastPreds,
          fusedPreds,
          actuals,
          weights,
        },
        null,
        2
      )
    );
    // Write TSV with formatted numbers (fixed width)
    const tsvHeader = [
      "index",
      "actual",
      "slowPred",
      "fastPred",
      "fusedPred",
      "weightSlow",
      "weightFast",
    ].join("\t");
    const tsvRows = actuals.map((actual, i) => [
      i,
      (actual ?? 0).toFixed(6),
      (slowPreds[i] ?? 0).toFixed(6),
      (fastPreds[i] ?? 0).toFixed(6),
      (fusedPreds[i] ?? 0).toFixed(6),
      (weights[i]?.[0] ?? 0).toFixed(6),
      (weights[i]?.[1] ?? 0).toFixed(6),
    ].join("\t"));
    const tsvContent = [tsvHeader, ...tsvRows].join("\n");
    fs.writeFileSync("./fixtures/kalman-filter-mae-comparison.tsv", tsvContent);
  });
});
