
#[macro_use]
#[allow(non_snake_case)]
pub mod ukf {
    extern crate nalgebra as na;
    use na::OMatrix;
    use na::{OVector, U15, U2, U3, U5, U7};

    use crate::sensor;
    use crate::sensor::measurement::DeviceSensor;
    use sensor::measurement::{
        HasSensorNoiseCovar, LidarMeasurement, RadarMeasurement, SensorMeasurement,
    };
    use sensor::measurement::{LidarSensor, RadarSensor};

    use crate::util::helper;
    use helper::negative_normalize;

    // U? are for nalegbra

    // state dimensions
    pub const N_X: usize = 5;
    pub type UX = U5;

    // augmented state dimensions
    pub const N_AUG: usize = 7;
    pub type UAUG = U7;

    // process noise standard deviation longitudinal acceleration in m/s^2
    pub const STD_A: f64 = 0.45;

    // process noise standard deviation yaw acceleration in rad/s^2
    pub const STD_YAWDD: f64 = 0.55;

    // size of lidar and radar measurement state vectors
    pub const N_Z_LIDAR: usize = 2;
    pub type UZLIDAR = U2;
    pub const N_Z_RADAR: usize = 3;
    pub type UZRADAR = U3;

    // sigma points (2 * N_AUG + 1)
    pub type USigmaPoints = U15;

    // define the spreading parameter
    pub const LAMBDA: f64 = 3.0 - N_AUG as f64;

    // state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
    pub type StateVector = OVector<f64, UX>;
    pub type AugStateVector = OVector<f64, UAUG>;

    pub type LidarStateVector = OVector<f64, UZLIDAR>;
    pub type RadarStateVector = OVector<f64, UZRADAR>;

    // measurement covariance matrices
    pub type CovarMatrix = OMatrix<f64, UX, UX>;
    pub type AugCovarMatrix = OMatrix<f64, UAUG, UAUG>;
    pub type CholeskyMatrix = AugCovarMatrix;
    pub type LidarCovarMatrix = OMatrix<f64, UZLIDAR, UZLIDAR>;
    pub type LidarNoiseCovarMatrix = LidarCovarMatrix;
    pub type RadarCovarMatrix = OMatrix<f64, UZRADAR, UZRADAR>;
    pub type RadarNoiseCovarMatrix = RadarCovarMatrix;

    // augmented sigma points matrix (n_aug, 2 * n_aug + 1)
    pub type AugSigmaPoints = OMatrix<f64, UAUG, USigmaPoints>;

    // predicted sigma points matrix (n_x, 2* n_aug + 1)
    pub type SigmaPoints = OMatrix<f64, UX, USigmaPoints>;
    pub type LidarSigmaPoints = OMatrix<f64, UZLIDAR, USigmaPoints>;
    pub type RadarSigmaPoints = OMatrix<f64, UZRADAR, USigmaPoints>;

    // sigma point weights
    pub type SigmaPointWeights = OVector<f64, USigmaPoints>;

    // cross correlation matrix
    pub type LidarCrossCorrelationMatrix = OMatrix<f64, UX, UZLIDAR>;
    pub type RadarCrossCorrelationMatrix = OMatrix<f64, UX, UZRADAR>;

    pub type LidarKalmanGain = OMatrix<f64, UX, UZLIDAR>;
    pub type RadarKalmanGain = OMatrix<f64, UX, UZRADAR>;

    pub trait HasStateVectorColumnSlice {
        fn state_from_col(&self, i: usize) -> StateVector;
    }

    impl HasStateVectorColumnSlice for SigmaPoints {
        fn state_from_col(&self, col: usize) -> StateVector {
            let mut x = StateVector::zeros();

            for i in 0..x.shape().0 {
                x[i] = self[(i, col)]
            }
            x
        }
    }

    pub trait KalmanGain<T> {}
    impl KalmanGain<LidarKalmanGain> for LidarKalmanGain {}
    impl KalmanGain<RadarKalmanGain> for RadarKalmanGain {}

    pub trait UnscentedKalmanUpdate<T, U, V>
    where
        T: KalmanGain<T>,
        U: SensorVector<U>,
        V: SensorCovar<V>,
    {
        fn update_state(x: &StateVector, K: &T, z_diff: &U) -> StateVector;
        fn update_covariance(P: &CovarMatrix, K: &T, S: &V) -> CovarMatrix;
    }

    impl UnscentedKalmanUpdate<LidarKalmanGain, LidarStateVector, LidarCovarMatrix> for LidarSensor {
        fn update_state(
            x: &StateVector,
            K: &LidarKalmanGain,
            z_diff: &LidarStateVector,
        ) -> StateVector {
            x + K * z_diff
        }
        fn update_covariance(
            P: &CovarMatrix,
            K: &LidarKalmanGain,
            S: &LidarCovarMatrix,
        ) -> CovarMatrix {
            P - K * S * K.transpose()
        }
    }

    impl UnscentedKalmanUpdate<RadarKalmanGain, RadarStateVector, RadarCovarMatrix> for RadarSensor {
        fn update_state(
            x: &StateVector,
            K: &RadarKalmanGain,
            z_diff: &RadarStateVector,
        ) -> StateVector {
            x + K * z_diff
        }
        fn update_covariance(
            P: &CovarMatrix,
            K: &RadarKalmanGain,
            S: &RadarCovarMatrix,
        ) -> CovarMatrix {
            P - K * S * K.transpose()
        }
    }

    pub trait SensorVectorType<T> {
        fn zeros() -> T;
    }
    impl SensorVectorType<LidarStateVector> for LidarStateVector {
        fn zeros() -> LidarStateVector {
            LidarStateVector::zeros()
        }
    }

    impl SensorVectorType<RadarStateVector> for RadarStateVector {
        fn zeros() -> RadarStateVector {
            RadarStateVector::zeros()
        }
    }

    pub trait SensorVector<T> {
        fn diff(&self, other: &T) -> T;
    }
    impl SensorVector<LidarStateVector> for LidarStateVector {
        fn diff(&self, other: &LidarStateVector) -> LidarStateVector {
            self.clone() - other
        }
    }

    impl SensorVector<RadarStateVector> for RadarStateVector {
        fn diff(&self, other: &RadarStateVector) -> RadarStateVector {
            self.clone() - other
        }
    }

    pub trait HasSensorVectorFactory<T, U>: SensorVector<T>
    where
        T: SensorVector<T>,
        U: SensorSigmaPoints<U>,
    {
        fn from_sensor_sigma_points(Z_sig: &U) -> T;
    }

    impl HasSensorVectorFactory<LidarStateVector, LidarSigmaPoints> for LidarStateVector {
        fn from_sensor_sigma_points(Z_sig: &LidarSigmaPoints) -> LidarStateVector {
            let mut z_pred = LidarStateVector::zeros();
            let weights = SigmaPointWeights::new();

            // mean predicted measurement
            for i in 0..Z_sig.cols() {
                z_pred += weights[i] * Z_sig.column(i);
            }
            z_pred
        }
    }

    impl HasSensorVectorFactory<RadarStateVector, RadarSigmaPoints> for RadarStateVector {
        fn from_sensor_sigma_points(Z_sig: &RadarSigmaPoints) -> RadarStateVector {
            let mut z_pred = RadarStateVector::zeros();
            let weights = SigmaPointWeights::new();

            // mean predicted measurement
            for i in 0..Z_sig.cols() {
                z_pred += weights[i] * Z_sig.column(i);
            }
            z_pred
        }
    }

    // convert measurements to state vectors
    pub trait HasMeasurementFactory<T, V, W>: SensorMeasurement<T>
    where
        T: SensorMeasurement<T>,
        V: SensorVector<V>,
        W: SensorCovar<W>,
    {
        fn z(&self) -> V;
        fn x(&self) -> V {
            self.z()
        }
        fn into_state(&self) -> V {
            self.z()
        }
        fn nis(&self, z_pred: &V, S: &W) -> f64;
    }

    impl HasMeasurementFactory<LidarMeasurement, LidarStateVector, LidarCovarMatrix>
        for LidarMeasurement
    {
        fn z(&self) -> LidarStateVector {
            LidarStateVector::new(self.px, self.py)
        }
        fn nis(&self, z_pred: &LidarStateVector, S: &LidarCovarMatrix) -> f64 {
            self.nis(&z_pred, &S)
        }
    }

    impl HasMeasurementFactory<RadarMeasurement, RadarStateVector, RadarCovarMatrix>
        for RadarMeasurement
    {
        fn z(&self) -> RadarStateVector {
            RadarStateVector::new(self.rho, self.theta, self.rho_dot)
        }
        fn nis(&self, z_pred: &RadarStateVector, S: &RadarCovarMatrix) -> f64 {
            self.nis(&z_pred, &S)
        }
    }

    // pub trait NewCovar<T> {
    //     fn new_covar(&self) -> T;
    // }

    // impl NewCovar<CovarMatrix> for CovarMatrix {
    //     fn new_covar(&self) -> CovarMatrix {
    //         CovarMatrix::zeros()
    //     }
    // }

    // impl NewCovar<LidarCovarMatrix> for LidarCovarMatrix {
    //     fn new_covar(&self) -> LidarCovarMatrix {
    //         LidarCovarMatrix::zeros()
    //     }
    // }

    // impl NewCovar<RadarCovarMatrix> for RadarCovarMatrix {
    //     fn new_covar(&self) -> RadarCovarMatrix {
    //         RadarCovarMatrix::zeros()
    //     }
    // }

    pub trait HasInverseCovar<T> {
        fn inverse(&self) -> T;
    }

    impl HasInverseCovar<LidarCovarMatrix> for LidarCovarMatrix {
        fn inverse(&self) -> LidarCovarMatrix {
            self.try_inverse().unwrap()
        }
    }

    impl HasInverseCovar<RadarCovarMatrix> for RadarCovarMatrix {
        fn inverse(&self) -> RadarCovarMatrix {
            self.try_inverse().unwrap()
        }
    }

    pub trait SensorCovar<T>: HasInverseCovar<T> {}
    impl SensorCovar<LidarCovarMatrix> for LidarCovarMatrix {}
    impl SensorCovar<RadarCovarMatrix> for RadarCovarMatrix {}

    pub trait SensorCovarType<T> {
        fn zeros() -> T;
    }
    impl SensorCovarType<LidarCovarMatrix> for LidarCovarMatrix {
        fn zeros() -> LidarCovarMatrix {
            LidarCovarMatrix::zeros()
        }
    }
    impl SensorCovarType<RadarCovarMatrix> for RadarCovarMatrix {
        fn zeros() -> RadarCovarMatrix {
            RadarCovarMatrix::zeros()
        }
    }

    pub trait HasSensorCovarFactory<T, U>
    where
        T: SensorCovar<T>,
        U: SensorSigmaPoints<U>,
    {
        fn from_sensor_sigma_points(Z_sig: &U) -> T;
    }

    impl HasSensorCovarFactory<LidarCovarMatrix, LidarSigmaPoints> for LidarCovarMatrix {
        fn from_sensor_sigma_points(Z_sig: &LidarSigmaPoints) -> LidarCovarMatrix {
            let mut S = LidarCovarMatrix::zeros();

            let weights = SigmaPointWeights::new();

            // measurement covariance matrix S
            for i in 1..Z_sig.cols() {
                let z_diff = Z_sig.column(i) - Z_sig.column(0);
                S += weights[i] * z_diff * z_diff.transpose();
            }

            // add measurement noise
            let lidar = LidarSensor::new();
            S += lidar.noise_covar_matrix();
            S
        }
    }

    pub trait HasSensorCovar<T> {
        fn measurement_covar(&self) -> T;
    }

    impl HasSensorCovar<LidarCovarMatrix> for LidarSigmaPoints {
        fn measurement_covar(&self) -> LidarCovarMatrix {
            LidarCovarMatrix::from_sensor_sigma_points(&self)
        }
    }

    impl HasSensorCovar<RadarCovarMatrix> for RadarSigmaPoints {
        fn measurement_covar(&self) -> RadarCovarMatrix {
            RadarCovarMatrix::from_sensor_sigma_points(&self)
        }
    }

    impl HasSensorCovarFactory<RadarCovarMatrix, RadarSigmaPoints> for RadarCovarMatrix {
        fn from_sensor_sigma_points(Z_sig: &RadarSigmaPoints) -> RadarCovarMatrix {
            let mut S = RadarCovarMatrix::zeros();

            let weights = SigmaPointWeights::new();

            // measurement covariance matrix S
            for i in 1..Z_sig.cols() {
                let z_diff = Z_sig.column(i) - Z_sig.column(0);
                S += weights[i] * z_diff * z_diff.transpose();
            }

            // add measurement noise
            let radar = RadarSensor::new();
            S += radar.noise_covar_matrix();
            S
        }
    }

    pub trait NewFromMeasurement<T, U> {
        fn new_from_measurement(m: T) -> U;
    }

    impl NewFromMeasurement<LidarMeasurement, LidarStateVector> for LidarStateVector {
        fn new_from_measurement(m: LidarMeasurement) -> LidarStateVector {
            LidarStateVector::new(m.px, m.py)
        }
    }

    impl NewFromMeasurement<RadarMeasurement, RadarStateVector> for RadarStateVector {
        fn new_from_measurement(m: RadarMeasurement) -> RadarStateVector {
            RadarStateVector::new(m.rho, m.theta, m.rho_dot)
        }
    }

    pub trait NCols {
        fn cols(&self) -> usize;
    }
    impl NCols for SigmaPoints {
        fn cols(&self) -> usize {
            self.shape().1
        }
    }
    impl NCols for LidarSigmaPoints {
        fn cols(&self) -> usize {
            self.shape().1
        }
    }
    impl NCols for RadarSigmaPoints {
        fn cols(&self) -> usize {
            self.shape().1
        }
    }

    pub trait NewSigmaPointWeights {
        fn new() -> SigmaPointWeights;
    }

    impl NewSigmaPointWeights for SigmaPointWeights {
        fn new() -> SigmaPointWeights {
            let n_aug = N_AUG;

            // create the sigma point weights (2*n_aug+1)
            let weight = 0.5 / (n_aug as f64 + LAMBDA);
            let mut weights = SigmaPointWeights::repeat(weight);
            weights[0] = LAMBDA / (LAMBDA + n_aug as f64);

            weights
        }
    }

    pub trait SensorSigmaPoints<T> {
        // fn from_sigma_points(X_sig_pred: &SigmaPoints) -> T;
    }
    impl SensorSigmaPoints<LidarSigmaPoints> for LidarSigmaPoints {
        // fn from_sigma_points(X_sig_pred: &SigmaPoints) -> LidarSigmaPoints {
        // X_sig_pred.measurement_space()
        // }
    }
    impl SensorSigmaPoints<RadarSigmaPoints> for RadarSigmaPoints {
        // fn from_sigma_points(X_sig_pred: &SigmaPoints) -> RadarSigmaPoints {
        // X_sig_pred.measurement_space()
        // }
    }

    pub trait HasSensorVector<T> {
        fn predicted_measurement(&self) -> T;
    }

    impl HasSensorVector<LidarStateVector> for LidarSigmaPoints {
        fn predicted_measurement(&self) -> LidarStateVector {
            LidarStateVector::from_sensor_sigma_points(&self)
        }
    }

    impl HasSensorVector<RadarStateVector> for RadarSigmaPoints {
        fn predicted_measurement(&self) -> RadarStateVector {
            RadarStateVector::from_sensor_sigma_points(&self)
        }
    }

    pub trait SensorSigmaPointsCrossCorrelation<T>: std::marker::Sized {}
    impl SensorSigmaPointsCrossCorrelation<LidarCrossCorrelationMatrix>
        for LidarCrossCorrelationMatrix
    {
    }
    impl SensorSigmaPointsCrossCorrelation<RadarCrossCorrelationMatrix>
        for RadarCrossCorrelationMatrix
    {
    }

    pub trait HasCrossCorrelationMatrix<T> {
        fn Tc(&self, X_sig_pred: &SigmaPoints) -> T;
    }

    impl HasCrossCorrelationMatrix<LidarCrossCorrelationMatrix> for LidarSigmaPoints {
        fn Tc(&self, X_sig_pred: &SigmaPoints) -> LidarCrossCorrelationMatrix {
            let weights = SigmaPointWeights::new();

            let mut Tc = LidarCrossCorrelationMatrix::zeros();
            let Z_sig = *self;
            let n_sig_cols = Z_sig.cols();

            for i in 1..n_sig_cols {
                let mut z_diff = Z_sig.column(i) - Z_sig.column(0);
                z_diff[1] = negative_normalize(z_diff[1]);
                let mut x_diff =
                    X_sig_pred.fixed_slice::<N_X, 1>(0, i) - X_sig_pred.fixed_slice::<N_X, 1>(0, 0);
                x_diff[3] = negative_normalize(x_diff[3]);
                Tc += weights[i] * x_diff * z_diff.transpose();
            }

            Tc
        }
    }

    impl HasCrossCorrelationMatrix<RadarCrossCorrelationMatrix> for RadarSigmaPoints {
        fn Tc(&self, X_sig_pred: &SigmaPoints) -> RadarCrossCorrelationMatrix {
            let weights = SigmaPointWeights::new();

            let mut Tc = RadarCrossCorrelationMatrix::zeros();
            let Z_sig = *self;
            let n_sig_cols = Z_sig.cols();

            for i in 1..n_sig_cols {
                let mut z_diff = Z_sig.column(i) - Z_sig.column(0);
                // TODO double check this normalisation in lidar
                z_diff[1] = negative_normalize(z_diff[1]);
                let mut x_diff =
                    X_sig_pred.fixed_slice::<N_X, 1>(0, i) - X_sig_pred.fixed_slice::<N_X, 1>(0, 0);
                x_diff[3] = negative_normalize(x_diff[3]);
                Tc += weights[i] * x_diff * z_diff.transpose();
            }

            Tc
        }
    }

    pub trait HasTcFactory<T, U, V> {
        fn mul(&self, rhs: U) -> V;
    }

    impl HasTcFactory<LidarCrossCorrelationMatrix, LidarCovarMatrix, LidarKalmanGain>
        for LidarCrossCorrelationMatrix
    {
        fn mul(&self, rhs: LidarCovarMatrix) -> LidarKalmanGain {
            (&self.clone() * rhs) as LidarKalmanGain
        }
    }

    impl HasTcFactory<RadarCrossCorrelationMatrix, RadarCovarMatrix, RadarKalmanGain>
        for RadarCrossCorrelationMatrix
    {
        fn mul(&self, rhs: RadarCovarMatrix) -> RadarKalmanGain {
            (&self.clone() * rhs) as RadarKalmanGain
        }
    }

    pub trait HasSensorSigmaPointsFactory<T, U>: HasCrossCorrelationMatrix<U> {}

    impl HasSensorSigmaPointsFactory<LidarSigmaPoints, LidarCrossCorrelationMatrix>
        for LidarSigmaPoints
    {
    }
    impl HasSensorSigmaPointsFactory<RadarSigmaPoints, RadarCrossCorrelationMatrix>
        for RadarSigmaPoints
    {
    }

    pub trait HasSigmaPointsMeasurementSpace<U> {
        fn measurement_space(&self) -> U;
    }

    impl HasSigmaPointsMeasurementSpace<LidarSigmaPoints> for SigmaPoints {
        fn measurement_space(&self) -> LidarSigmaPoints {
            let X_sig_pred = *self as SigmaPoints;
            let mut Z_sig = LidarSigmaPoints::zeros();

            for i in 0..Z_sig.cols() {
                let p_x = X_sig_pred[(0, i)];
                let p_y = X_sig_pred[(1, i)];

                // measurement model
                Z_sig[(0, i)] = p_x;
                Z_sig[(1, i)] = p_y;
            }

            Z_sig
        }
    }

    impl HasSigmaPointsMeasurementSpace<RadarSigmaPoints> for SigmaPoints {
        fn measurement_space(&self) -> RadarSigmaPoints {
            let X_sig_pred = *self as SigmaPoints;
            let mut Z_sig = RadarSigmaPoints::zeros();

            for i in 0..Z_sig.cols() {
                let p_x = X_sig_pred[(0, i)];
                let p_y = X_sig_pred[(1, i)];
                let v = X_sig_pred[(2, i)];
                let yaw = X_sig_pred[(3, i)];
                let v1 = v * yaw.cos();
                let v2 = v * yaw.sin();

                // measurement model
                Z_sig[(0, i)] = (p_x * p_x + p_y * p_y).sqrt();
                Z_sig[(1, i)] = p_y.atan2(p_x);
                if Z_sig[(0, i)] != 0.0 {
                    Z_sig[(2, i)] = (p_x * v1 + p_y * v2) / Z_sig.column(i)[0]; // r_dot
                } else {
                    Z_sig[(2, i)] = 0.0;
                }
            }

            Z_sig
        }
    }
}
