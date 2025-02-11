import { expect } from 'chai';
import { KalmanFilter } from './kalman-filter';

describe('KalmanFilter', function () {
  let kf: KalmanFilter;

  beforeEach(function () {
    kf = new KalmanFilter(1, 1);
  });

  it('should initialize the Kalman filter', function () {
    kf.init(0);
    expect(kf.kf).to.not.be.undefined;
    expect(kf.lastTS).to.be.a('number');
  });

  it('should predict the next state', function () {
    kf.init(0);
    const steptime = kf.predict();
    expect(steptime).to.be.a('number');
    expect(kf.state).to.not.be.null;
  });

  it('should correct the state with a new observation', function () {
    kf.init(0);
    kf.predict();
    kf.correct(1, 1);
    expect(kf.state).to.not.be.null;
  });

  it('should return the mean of the state', function () {
    kf.init(0);
    kf.predict();
    kf.correct(1, 1);
    const mean = kf.mean();
    expect(mean).to.be.an('array').that.has.lengthOf(2);
  });

  it('should return the count of the state updates', function () {
    kf.init(0);
    kf.predict();
    kf.correct(1, 1);
    const count = kf.count();
    expect(count).to.be.a('number');
  });
});