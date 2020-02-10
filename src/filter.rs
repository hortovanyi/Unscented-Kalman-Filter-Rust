pub mod kalman_filter {

  extern crate nalgebra as na;
  use na::{DMatrix, MatrixN, MatrixMN, DVector, VectorN, Vector2, U5, U7, U15};

  use na::Cholesky;

  use crate::sensor;
  use sensor::measurement::{LidarMeasurement, RadarMeasurement, MeasurementPackage, SensorType};

  // state dimensions
  const N_X: usize = 5;

  // augmented state dimansions
  const N_AUG: usize = 7;

  // process noise standard deviation longitudinal acceleration in m/s^2
  const STD_A: f64 = 0.45;

  // process noise standard deviation yaw acceleration in rad/s^2
  const STD_YAWDD: f64 = 0.55;

  // state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  type StateVector = VectorN<f64, U5>;
  type AugStateVector = VectorN<f64, U7>;

  // measurement covariance matrices
  type CovarMatrix = MatrixN<f64, U5>;
  type AugCovarMatrix = MatrixN<f64, U7>;
  type CholeskyMatrix = AugCovarMatrix;

  // augmented sigma points matrix (n_aug, 2 * n_aug + 1)
  type AugSigmaPoints = MatrixMN<f64, U7, U15>;

  // predicted sigma points matrix (n_x, 2* n_aug + 1)
  type SigmaPoints = MatrixMN<f64, U5, U15>;

  // sigma point weights
  type SigmaPointWeights = VectorN<f64, U15>;

  // weights vector
  type WeightsVector = DVector<f64>;

  // cross correlation matrix
  type CrossCorrelationMatrix = DMatrix<f64>;

  #[allow(non_snake_case)]
  #[derive(Debug)]
  pub struct UnscentedKalmanFilter {

    // initially set to false, set to ture in frist call of ProcessMeasurement
    is_initiliased: bool,

    // if this is false, lidar measurements will be ignores (except for init)
    use_laser: bool,

    // if this is false, radar measurements will be ignored (except for init)
    use_radar: bool,

    // previous timestamp
    prev_timestamp: u64,

    // Covariance Matrix
    P: CovarMatrix,

    // StateVector
    x: StateVector,
  }

  #[allow(non_snake_case)]
  trait UKFPredict {
    fn prediction(x:StateVector, P:CovarMatrix, delta_t: f64) -> (SigmaPoints, StateVector, CovarMatrix){
      let X_sig_aug = Self::augmented_sigma_points(N_X, N_AUG, STD_A, STD_YAWDD, x, P);
      let X_sig_pred = Self::predict_sigma_points(N_X, N_AUG, delta_t, X_sig_aug);
      Self::predict_mean_and_covar(N_X, N_AUG, X_sig_pred)
    }

    fn augmented_sigma_points(n_x: usize, n_aug: usize, std_a: f64, std_yawdd: f64, x:StateVector, P:CovarMatrix) -> AugSigmaPoints{
      // create augmented state vector
      let mut x_aug = AugStateVector::zeros();
      // x_aug.fixed_rows_mut::<U5>(0).copy_from(&x); 
      x_aug.copy_from(&x.clone().resize(n_aug,1,0.0));
      debug!("{:?} {:?}", x_aug, x);


      // create augmented coveriance matrix
      let mut P_aug = AugCovarMatrix::zeros();
      P_aug.copy_from(&P.clone().resize(n_aug, n_aug, 0.0));
      P_aug[(5,5)] = std_a * std_a;
      P_aug[(6,6)] = std_yawdd * std_yawdd;

      debug!("{:?} {:?}", P_aug, P );

      // square root of P
      let L:CholeskyMatrix = match P_aug.cholesky() {
          Some(x) => x.l(),
          None => {warn!("no_cholesky!"); CholeskyMatrix::zeros() },
      };

      // define spreading parameter
      let lambda:f64 = 3.0 - n_aug as f64;
      let spread = (lambda as f64 + n_aug as f64).sqrt();   
      
      let X_sig_aug = AugSigmaPoints::zeros();

      X_sig_aug.column_mut(0).copy_from(&x_aug);

      for i in 0 .. n_aug {
        let mut x_aug1_col = x_aug.clone();
        let mut x_aug2_col = x_aug.clone();
        for j in 0 .. n_aug {
          x_aug1_col[j] += spread * L.column(i)[j];
          x_aug2_col[j] -= spread * L.column(i)[j];
        }
        X_sig_aug.column_mut(i+1).copy_from(&x_aug1_col);
        X_sig_aug.column_mut(i+1+n_aug).copy_from(&x_aug2_col);
      }


      X_sig_aug
    }

    fn predict_sigma_points(n_x: usize, n_aug: usize, delta_t: f64, X_sig_aug: AugSigmaPoints) -> SigmaPoints{
      let mut X_sig_pred = SigmaPoints::zeros();

      for i in 0 .. 2 * n_aug + 1 {
        // extract values for better readability
        let p_x = X_sig_aug[(0,i)];
        let p_y = X_sig_aug[(1,i)];
        let v = X_sig_aug[(2,i)];
        let yaw = X_sig_aug[(3,i)];
        let yawd = X_sig_aug[(4,i)];
        let nu_a = X_sig_aug[(5,i)];
        let nu_yawdd = X_sig_aug[(6,i)];

        // predicted state values in *_p 
        let mut px_p:f64 = 0.0;
        let mut py_p:f64 = 0.0;

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

    fn predict_mean_and_covar(n_x: usize, n_aug: usize, X_sig_pred:SigmaPoints) -> (SigmaPoints, StateVector, CovarMatrix){
      // create the predicted state vector
      let mut x = StateVector::zeros();

      // create the predicted covariance matrix
      let mut P = CovarMatrix::zeros();

      // define spreading parameter
      let lambda:f64 = 3.0 - n_aug as f64;
      
      // create the sigma point weights (2*n_aug+1)
      let weight = 0.5 / (n_aug as f64 + lambda);
      let mut weights = SigmaPointWeights::repeat(weight);
      weights[0] = lambda / (lambda+n_aug as f64);
      
      (X_sig_pred, x, P)
    }
  }

  #[allow(non_snake_case)]
  trait UKFUpdate {
    fn predict_measurement(n_x: usize, n_aug:usize, n_z: usize, Z_sig:SigmaPoints, X_sig_pred:SigmaPoints) -> (StateVector, CovarMatrix);
    fn mean_predicted_measurement(n_z: usize, n_aug: usize, weights:WeightsVector, Z_sig:SigmaPoints) -> StateVector;
    fn measurement_covar_matrix_s(n_z: usize, n_aug: usize, Z_sig:SigmaPoints, z_pred:StateVector, weights:WeightsVector) -> CovarMatrix;
    fn update_state(n_x:usize, n_aug:usize, n_z:usize, X_sig_pred:SigmaPoints, x:StateVector, P:CovarMatrix,
                    Z_sig:SigmaPoints, z_pred:StateVector, S:CovarMatrix, z:StateVector) -> (StateVector, CovarMatrix);
    fn cross_correlation_matrix(n_x:usize, n_aug:usize, n_z:usize, Z_sig:SigmaPoints, Z_pred:StateVector, X_sig_pred:SigmaPoints,
                                x:StateVector, weights:WeightsVector) -> CrossCorrelationMatrix;
  }

  #[allow(non_snake_case)]
  trait UKFLidarUpdate: UKFUpdate {
    fn lidar_update(measurement_package: MeasurementPackage,  X_sig_pred: SigmaPoints,
                  x: StateVector, P: CovarMatrix) -> (StateVector, CovarMatrix);
    fn predict_lidar_measurement(n_x: usize, n_aug: usize, n_z: usize, X_sig_pred: SigmaPoints) -> (StateVector, CovarMatrix);
    fn lidar_noise_covar_matrix() -> CovarMatrix;
    fn sigma_points_lidar_measurement_space(n_z: usize, n_aug: usize, X_sig_pred:SigmaPoints) -> SigmaPoints;
  }

  #[allow(non_snake_case)]
  trait UKFRadarUpdate: UKFUpdate {
    fn radar_update(measurement_package: MeasurementPackage,  X_sig_pred: SigmaPoints,
                  x: StateVector, P: CovarMatrix) -> (StateVector, CovarMatrix);
    fn predict_radar_measurement(n_x: usize, n_aug: usize, n_z: usize, X_sig_pred: SigmaPoints) -> (StateVector, CovarMatrix);
    fn radar_noise_covar_matrix() -> CovarMatrix;
    fn sigma_points_radar_measurement_space(n_z: usize, n_aug: usize, X_sig_pred:SigmaPoints) -> SigmaPoints;
  }

  // trait UKF: UKFPredict + UKFLidarUpdate + UKFRadarUpdate{
  pub trait UKF: UKFPredict{
    fn new() -> Self;
    fn init_state_lidar(m:&LidarMeasurement) -> StateVector;
    fn init_state_radar(m:&RadarMeasurement) -> StateVector;
    fn initialise(&mut self, m:&MeasurementPackage);
    fn process_measurement(&mut self, m:&MeasurementPackage) -> StateVector;
  }

  #[allow(non_snake_case, dead_code)]
  impl UKF for UnscentedKalmanFilter {
    fn new() -> UnscentedKalmanFilter {
      let P = CovarMatrix::from_diagonal_element(1.0);
      let x = StateVector::zeros();

      UnscentedKalmanFilter{is_initiliased:false, 
                            use_laser:true, 
                            use_radar:true, 
                            prev_timestamp:0, 
                            P:P, 
                            x:x} 
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

      x[0]=px;
      x[1]=py;
      x[2]=v;
      x[3]=yaw;
      x[4]=yaw_dot;
      
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

      debug!("init x:{:?}", self.x);
      debug!("init P:{:?}", self.P); 
    }

    #[allow(dead_code)]
    fn process_measurement(&mut self, m:&MeasurementPackage) -> StateVector {
      // initialisation
      if !self.is_initiliased {
        self.initialise(m);
        return self.x;
      }

      let delta_t:f64 = (m.timestamp - self.prev_timestamp) as f64 / 100000.0;
      self.prev_timestamp = m.timestamp;



      self.x
    }
  }

}