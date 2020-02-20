pub mod kalman_filter {

    extern crate nalgebra as na;
    // use na::Scalar;
    // use na::{DVector, VectorN, U1, U15, U2, U3, U5, U7};

    use crate::sensor;
    use sensor::measurement::{DeviceSensor, SensorMeasurement};
    use sensor::measurement::{LidarMeasurement, LidarSensor, RadarMeasurement, RadarSensor};
    use sensor::measurement::{MeasurementPackage, SensorType};

    use crate::ukf_type;
    use ukf_type::ukf::*;

    use crate::util::helper;
    use helper::negative_normalize;

    #[allow(non_snake_case)]
    #[derive(Debug)]
    pub struct UnscentedKalmanFilter {
        // initially set to false, set to ture in frist call of ProcessMeasurement
        is_initiliased: bool,

        // if this is false, lidar measurements will be ignores (except for init)
        use_laser: bool,

        // if this is false, radar measurements will be ignored (except for init)
        use_radar: bool,

        lidar_sensor: LidarSensor,
        radar_sensor: RadarSensor,

        // previous timestamp
        prev_timestamp: u64,

        // Covariance Matrix
        P: CovarMatrix,

        // StateVector
        x: StateVector,

        // Normalised Innovation Squared
        pub nis_lidar: f64,
        pub nis_radar: f64,
    }

    #[allow(non_snake_case)]
    pub trait UKFPredict {
        fn prediction(
            x: &StateVector,
            P: &CovarMatrix,
            delta_t: f64,
        ) -> (SigmaPoints, StateVector, CovarMatrix) {
            let X_sig_aug = Self::augmented_sigma_points(STD_A, STD_YAWDD, &x, &P);
            let X_sig_pred = Self::predict_sigma_points(delta_t, &X_sig_aug);
            Self::predict_mean_and_covar(X_sig_pred)
        }

        fn augmented_sigma_points(
            std_a: f64,
            std_yawdd: f64,
            x: &StateVector,
            P: &CovarMatrix,
        ) -> AugSigmaPoints {
            // create augmented state vector
            let mut x_aug = AugStateVector::zeros();
            // x_aug.fixed_rows_mut::<U5>(0).copy_from(&x);
            let n_aug = x_aug.shape().0;
            x_aug.copy_from(&x.clone().resize(n_aug, 1, 0.0));

            // create augmented coveriance matrix
            let mut P_aug = AugCovarMatrix::zeros();
            P_aug.copy_from(&P.clone().resize(n_aug, n_aug, 0.0));
            P_aug[(5, 5)] = std_a * std_a;
            P_aug[(6, 6)] = std_yawdd * std_yawdd;

            // square root of P
            let L: CholeskyMatrix = match P_aug.cholesky() {
                Some(x) => x.l(),
                None => {
                    warn!("no_cholesky!");
                    CholeskyMatrix::zeros()
                }
            };

            // define spreading parameter
            let spread = (LAMBDA + n_aug as f64).sqrt();

            let mut X_sig_aug = AugSigmaPoints::zeros();

            X_sig_aug.column_mut(0).copy_from(&x_aug);

            for i in 0..n_aug {
                let mut x_aug1_col = x_aug.clone();
                let mut x_aug2_col = x_aug.clone();
                for j in 0..n_aug {
                    x_aug1_col[j] += spread * L.column(i)[j];
                    x_aug2_col[j] -= spread * L.column(i)[j];
                }
                X_sig_aug.column_mut(i + 1).copy_from(&x_aug1_col);
                X_sig_aug.column_mut(i + 1 + n_aug).copy_from(&x_aug2_col);
            }

            X_sig_aug
        }

        fn predict_sigma_points(delta_t: f64, X_sig_aug: &AugSigmaPoints) -> SigmaPoints {
            let mut X_sig_pred = SigmaPoints::zeros();

            let n_aug = X_sig_aug.shape().0;

            for i in 0..2 * n_aug + 1 {
                // extract values for better readability
                let p_x = X_sig_aug[(0, i)];
                let p_y = X_sig_aug[(1, i)];
                let v = X_sig_aug[(2, i)];
                let yaw = X_sig_aug[(3, i)];
                let yawd = X_sig_aug[(4, i)];
                let nu_a = X_sig_aug[(5, i)];
                let nu_yawdd = X_sig_aug[(6, i)];

                // predicted state values in *_p
                let mut px_p: f64 = 0.0;
                let mut py_p: f64 = 0.0;

                // avoid division by zero
                if yawd.is_normal() {
                    px_p = p_x + v / yawd * ((yaw + yawd * delta_t).sin() - yaw.sin());
                    py_p = p_y + v / yawd * (yaw.cos() - (yaw + yawd * delta_t).cos());
                } else {
                    px_p = p_x + v * delta_t * yaw.cos();
                    py_p = p_y + v * delta_t * yaw.sin();
                }

                let mut v_p = v;
                let mut yaw_p = yaw + yawd * delta_t;
                let mut yawd_p = yawd;

                // add noise
                px_p = px_p + 0.5 * nu_a * delta_t * delta_t * yaw.cos();
                py_p = py_p + 0.5 * nu_a * delta_t * delta_t * yaw.sin();
                v_p = v_p + nu_a * delta_t;

                yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
                yawd_p = yawd_p + nu_yawdd * delta_t;

                //write predicted sigma point into right column
                X_sig_pred[(0, i)] = px_p;
                X_sig_pred[(1, i)] = py_p;
                X_sig_pred[(2, i)] = v_p;
                X_sig_pred[(3, i)] = yaw_p;
                X_sig_pred[(4, i)] = yawd_p;
            }

            X_sig_pred
        }

        fn predict_mean_and_covar(
            X_sig_pred: SigmaPoints,
        ) -> (SigmaPoints, StateVector, CovarMatrix) {
            // create the predicted state vector
            let mut x = StateVector::zeros();

            // create the predicted covariance matrix
            let mut P = CovarMatrix::zeros();

            // create the sigme point weights
            let weights = SigmaPointWeights::new();

            let n_sig_cols = X_sig_pred.cols();

            // predicted state mean
            for i in 0..n_sig_cols {
                x += weights[i] * X_sig_pred.state_from_col(i);
            }

            // predicted state covariance matrix
            for i in 1..n_sig_cols {
                // state difference
                let mut x_diff = X_sig_pred.state_from_col(i) - X_sig_pred.state_from_col(0);

                // angle normalization -pi to pi
                x_diff[3] = negative_normalize(x_diff[3]);

                P += weights[i] * x_diff * x_diff.transpose();
            }

            (X_sig_pred, x, P)
        }
    }

    #[allow(non_snake_case)]
    trait UKFUpdate<T, U, V, W, X, Y>: UnscentedKalmanUpdate<Y, V, W>
    where
        T: SensorMeasurement<T> + HasMeasurementFactory<T, V, W>,
        U: SensorSigmaPoints<U>
            + HasSensorSigmaPointsFactory<U, X>
            + HasSensorVector<V>
            + HasSensorCovar<W>,
        V: SensorVector<V> + HasSensorVectorFactory<V, U>,
        W: SensorCovar<W> + HasSensorCovarFactory<W, U>,
        X: SensorSigmaPointsCrossCorrelation<X> + HasTcFactory<X, W, Y>,
        Y: KalmanGain<Y>,
    {
        fn update(
            &self,
            m: &T,
            X_sig_pred: &SigmaPoints,
            x: &StateVector,
            P: &CovarMatrix,
        ) -> (StateVector, CovarMatrix, f64)
        where
            SigmaPoints: HasSigmaPointsMeasurementSpace<U>,
        {
            // measurement
            let z = m.z();

            // Predict next measurement

            // Get the specific devices sigma points
            // let Z_sig:U = SensorSigmaPoints::<U>::from_sigma_points(&X_sig_pred);
            let Z_sig = X_sig_pred.measurement_space();

            // mean predicted measurement
            let z_pred = Z_sig.predicted_measurement();

            // measurement covariance matrix S
            let S = Z_sig.measurement_covar();

            // Kalman Update

            // cross correlation matric
            let Tc = Z_sig.Tc(&X_sig_pred);

            // Kalman gain K
            let K = Tc.mul(S.inverse());

            // residual
            let z_diff = z.diff(&z_pred);

            // update state and covariance
            // let x = x + K * z_diff;
            // let P = P - K * S * K.transpose();
            let x = Self::update_state(&x, &K, &z_diff);
            let P = Self::update_covariance(&P, &K, &S);

            let nis = m.nis(&z_pred, &S);

            (x, P, nis)
        }
    }

    impl
        UKFUpdate<
            LidarMeasurement,
            LidarSigmaPoints,
            LidarStateVector,
            LidarCovarMatrix,
            LidarCrossCorrelationMatrix,
            LidarKalmanGain,
        > for LidarSensor
    {
        // fn update(
        //     &self,
        //     m: LidarMeasurement,
        //     X_sig_pred: &SigmaPoints,
        //     x: &StateVector,
        //     P: &CovarMatrix,
        // ) -> (StateVector, CovarMatrix) {
        //     // measurement
        //     let z = LidarStateVector::new_from_measurement(m);
        //     let n_z = z.shape().0;
        //     assert!(n_z == self.n_z, "n_z != {} for lidar", self.n_z);

        //     let (Z_sig, z_pred, S) = self.predict_measurement(&X_sig_pred);

        //     // cross correlation matric
        //     let Tc = Z_sig.Tc(&X_sig_pred);

        //     // Kalman gain K
        //     let K = Tc * S.try_inverse().unwrap();

        //     // residual
        //     let z_diff = z - z_pred;

        //     // update state and covariance
        //     let x = x + K * z_diff;
        //     let P = P - K * S * K.transpose();

        //     (x, P)
        // }

        //     fn predict_measurement(&self, X_sig_pred: &SigmaPoints) -> (LidarSigmaPoints, LidarStateVector, LidarCovarMatrix) {
        //         let mut z_Pred = LidarStateVector::new_state();
        //         let mut S = LidarCovarMatrix::new_covar();
        //         let Z_sig:LidarSigmaPoints = X_sig_pred.measurement_space();

        //         let weights = SigmaPointWeights::new();

        //         // mean predicted measurement
        //         for i in 0 .. Z_sig.cols() {
        //             z_Pred += weights[i] * Z_sig.column(i);
        //         }

        //         // measurement covariance matrix S
        //         for i in 1 .. Z_sig.cols() {
        //             let z_diff = Z_sig.column(i) - Z_sig.column(0);
        //             S += weights[i] * z_diff * z_diff.transpose();
        //         }

        //         // add measurement noise
        //         S += self.noise_covar_matrix();

        //         (Z_sig, z_Pred, S)
        //     }
    }
    impl
        UKFUpdate<
            RadarMeasurement,
            RadarSigmaPoints,
            RadarStateVector,
            RadarCovarMatrix,
            RadarCrossCorrelationMatrix,
            RadarKalmanGain,
        > for RadarSensor
    {
    }

    // trait UKF: UKFPredict + UKFLidarUpdate + UKFRadarUpdate{

    impl UKFPredict for UnscentedKalmanFilter {}

    pub trait UKF: UKFPredict {
        fn new() -> Self;
        fn init_state_lidar(m: &LidarMeasurement) -> StateVector;
        fn init_state_radar(m: &RadarMeasurement) -> StateVector;
        fn initialise(&mut self, m: &MeasurementPackage);
        fn process_measurement(&mut self, m: &MeasurementPackage) -> StateVector;
    }

    #[allow(non_snake_case, dead_code)]
    impl UKF for UnscentedKalmanFilter {
        fn new() -> UnscentedKalmanFilter {
            let P = CovarMatrix::from_diagonal_element(1.0);
            let x = StateVector::zeros();

            UnscentedKalmanFilter {
                is_initiliased: false,
                use_laser: true,
                use_radar: true,
                lidar_sensor: LidarSensor::new(),
                radar_sensor: RadarSensor::new(),
                prev_timestamp: 0,
                P: P,
                x: x,
                nis_lidar: 0.0,
                nis_radar: 0.0,
            }
        }

        fn init_state_lidar(m: &LidarMeasurement) -> StateVector {
            let mut x = StateVector::zeros();
            x[0] = m.px;
            x[1] = m.py;

            trace!("init_state_lidar x:{:?}", x);
            x
        }

        fn init_state_radar(m: &RadarMeasurement) -> StateVector {
            let mut x = StateVector::zeros();

            let rho = m.rho; // Range - radial distance from origin
            let phi = m.theta; // bearing - angle between rho and x
            let rho_dot = m.rho_dot; // Radial Velocity - change of p(range rate)

            let px = rho * phi.cos(); // metres
            let py = rho * phi.sin();
            let v = rho_dot; // metres/sec
            let yaw = phi; // radians
            let yaw_dot = 0.0; // radians/sec

            x[0] = px;
            x[1] = py;
            x[2] = v;
            x[3] = yaw;
            x[4] = yaw_dot;

            trace!("init_state_radar x:{:?}", x);
            x
        }

        fn initialise(&mut self, m: &MeasurementPackage) {
            self.x = match m.sensor_type {
                SensorType::Lidar => Self::init_state_lidar(m.lidar_data.as_ref().unwrap()),
                SensorType::Radar => Self::init_state_radar(m.radar_data.as_ref().unwrap()),
            };

            self.prev_timestamp = m.timestamp;

            self.is_initiliased = true;

            debug!("init x:{}", self.x);
            debug!("init P:{}", self.P);

            debug!("init lidar_sensor:{:?}", self.lidar_sensor);
            debug!("init radar_sensor:{:?}", self.radar_sensor);
        }

        fn process_measurement(&mut self, m: &MeasurementPackage) -> StateVector {
            // initialisation
            if !self.is_initiliased {
                self.initialise(m);
                return self.x;
            }

            let delta_t: f64 = (m.timestamp - self.prev_timestamp) as f64 / 1000000.0;
            self.prev_timestamp = m.timestamp;

            // prediction step
            let (X_sig_pred, x, P) = Self::prediction(&self.x, &self.P, delta_t);

            // update step
            let (x, P, nis) = match m.sensor_type {
                SensorType::Lidar => {
                    self.lidar_sensor
                        .update(m.lidar_data.as_ref().unwrap(), &X_sig_pred, &x, &P)
                }
                SensorType::Radar => {
                    self.radar_sensor
                        .update(m.radar_data.as_ref().unwrap(), &X_sig_pred, &x, &P)
                }
            };

            match m.sensor_type {
                SensorType::Lidar => self.nis_lidar = nis,
                SensorType::Radar => self.nis_radar = nis,
            }
            self.x = x;
            self.P = P;

            self.x
        }
    }
}
