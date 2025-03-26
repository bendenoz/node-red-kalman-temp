import { KalmanTempNodeDef } from "./kalman-temp";
import { KalmanFilter } from "./kalman-filter"; // Mocked in tests

jest.mock("./kalman-filter", () => {
  return {
    KalmanFilter: jest.fn().mockImplementation(() => ({
      init: jest.fn(),
      predict: jest.fn(() => 1),
      correct: jest.fn(),
      mean: jest.fn(() => [42]),
    })),
  };
});

describe("KalmanTempNode", () => {
  let RED: any;
  let node: any;

  beforeEach(() => {
    RED = {
      nodes: {
        createNode: jest.fn(),
        registerType: jest.fn(),
      },
    };
  });

  it("should register the node type", () => {
    const module = require("./kalman-temp"); // Import module
    module(RED);
    expect(RED.nodes.registerType).toHaveBeenCalledWith(
      "kalman-temp",
      expect.any(Function)
    );
  });

  it("should initialize the Kalman filter on first input", () => {
    const module = require("./kalman-temp");
    module(RED);

    const config: Partial<KalmanTempNodeDef> = {
      id: "1",
      type: "kalman-temp",
      name: "test",
      R: 0.2,
      Q: 0.001,
      predictInterval: 60,
    };
    const KalmanTempNode = RED.nodes.registerType.mock.calls[0][1];

    node = { on: jest.fn(), send: jest.fn(), warn: jest.fn() };
    KalmanTempNode.call(node, config);

    const inputCallback = node.on.mock.calls.find(
      ([event]: [string, () => void]) => event === "input"
    )[1];
    inputCallback({ payload: 25 });
    const closeCallback: () => void = node.on.mock.calls.find(
      ([event]: [string, () => void]) => event === "close"
    )[1];
    closeCallback();

    expect(KalmanFilter).toHaveBeenCalledWith(0.2, 0.001);
    expect(node.send).toHaveBeenCalledWith([{ payload: 42 }]);
  });

  it("should handle non-numeric input gracefully", () => {
    const module = require("./kalman-temp");
    module(RED);

    const config: Partial<KalmanTempNodeDef> = {
      id: "1",
      type: "kalman-temp",
      name: "test",
      R: 0.2,
      Q: 0.001,
      predictInterval: 60,
    };
    const KalmanTempNode = RED.nodes.registerType.mock.calls[0][1];

    node = { on: jest.fn(), send: jest.fn(), warn: jest.fn() };
    KalmanTempNode.call(node, config);

    const inputCallback = node.on.mock.calls.find(
      ([event]: [string, () => void]) => event === "input"
    )[1];
    inputCallback({ payload: "invalid" });
    const closeCallback: () => void = node.on.mock.calls.find(
      ([event]: [string, () => void]) => event === "close"
    )[1];
    closeCallback();

    expect(node.warn).toHaveBeenCalledWith("Input must be a valid number.");
    expect(node.send).not.toHaveBeenCalled();
  });

  it("should predict values periodically", () => {
    jest.useFakeTimers();
    const module = require("./kalman-temp");
    module(RED);

    const config: Partial<KalmanTempNodeDef> = {
      id: "1",
      type: "kalman-temp",
      name: "test",
      R: 0.2,
      Q: 0.001,
      predictInterval: 60,
    };
    const KalmanTempNode = RED.nodes.registerType.mock.calls[0][1];

    node = { on: jest.fn(), send: jest.fn(), warn: jest.fn() };
    KalmanTempNode.call(node, config);

    const inputCallback = node.on.mock.calls.find(
      ([event]: [string, () => void]) => event === "input"
    )[1];
    inputCallback({ payload: 30 });

    jest.advanceTimersByTime(61000);

    const closeCallback: () => void = node.on.mock.calls.find(
      ([event]: [string, () => void]) => event === "close"
    )[1];
    closeCallback();

    expect(node.send).toHaveBeenCalledTimes(2); // One from input, one from prediction
  });

  it("should clear timeout on node close", () => {
    jest.useFakeTimers();
    const module = require("./kalman-temp");
    module(RED);

    const config: Partial<KalmanTempNodeDef> = {
      id: "1",
      type: "kalman-temp",
      name: "test",
      R: 0.2,
      Q: 0.001,
      predictInterval: 60,
    };
    const KalmanTempNode = RED.nodes.registerType.mock.calls[0][1];

    node = { on: jest.fn(), send: jest.fn(), warn: jest.fn() };
    KalmanTempNode.call(node, config);

    const closeCallback: () => void = node.on.mock.calls.find(
      ([event]: [string, () => void]) => event === "close"
    )[1];
    closeCallback();

    jest.advanceTimersByTime(61000);
    expect(node.send).toHaveBeenCalledTimes(0);
  });
});
