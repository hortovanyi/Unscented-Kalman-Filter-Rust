pub mod measurement {

    extern crate nalgebra as na;

    use na::{Vector2, Vector3, Vector4};

    use crate::ukf_type;
    use ukf_type::ukf::*;

    pub trait SensorMeasurement<T> {}

    pub trait DeviceSensor<T> {
        fn new() -> T;
    }
    pub trait HasSensorNoiseCovar<T, U>: DeviceSensor<T> {
        fn noise_covar_matrix(&self) -> U;
    }

    #[derive(Debug)]
    pub struct LidarMeasurement {
        pub px: f64,
        pub py: f64,
    }

    impl LidarMeasurement {
        pub fn point(&self) -> (f64, f64) {
            (self.px, self.py)
        }
        // Normalised Innovation Squared
        pub fn nis(&self, z_pred: &LidarStateVector, S: &LidarCovarMatrix) -> f64 {
            let raw_v = Vector2::<f64>::new(self.px, self.py);
            let z_pred = Vector2::<f64>::new(z_pred[0], z_pred[1]);

            let diff_pred = raw_v - z_pred;
            let S_inv = S.try_inverse().unwrap();

            let nis = diff_pred.transpose() * S_inv * diff_pred;

            nis[(0, 0)]
        }
    }

    impl SensorMeasurement<LidarMeasurement> for LidarMeasurement {}

    #[derive(Debug)]
    pub struct LidarSensor {
        pub n_z: usize,     // no of measurements
        pub std_laspx: f64, // noise standard deviation px in m
        pub std_laspy: f64, // noise standard deviation py in m
    }

    impl DeviceSensor<LidarSensor> for LidarSensor {
        fn new() -> LidarSensor {
            LidarSensor {
                n_z: N_Z_LIDAR,
                std_laspx: 0.15,
                std_laspy: 0.15,
            }
        }
    }

    impl HasSensorNoiseCovar<LidarSensor, LidarNoiseCovarMatrix> for LidarSensor {
        fn noise_covar_matrix(&self) -> LidarNoiseCovarMatrix {
            LidarNoiseCovarMatrix::new(
                self.std_laspx * self.std_laspx,
                0.0,
                0.0,
                self.std_laspy * self.std_laspy,
            )
        }
    }

    #[derive(Debug)]
    pub struct RadarMeasurement {
        pub rho: f64,
        pub theta: f64,
        pub rho_dot: f64,
    }

    impl RadarMeasurement {
        pub fn point(&self) -> (f64, f64) {
            (self.rho * self.theta.cos(), self.rho * self.theta.sin())
        }
        // Normalised Innovation Squared
        pub fn nis(&self, z_pred: &RadarStateVector, S: &RadarCovarMatrix) -> f64 {
            let raw_v = Vector3::<f64>::new(self.rho, self.theta, self.rho_dot);
            let z_pred = Vector3::<f64>::new(z_pred[0], z_pred[1], z_pred[2]);

            let diff_pred = raw_v - z_pred;
            let S_inv = S.try_inverse().unwrap();

            let nis = diff_pred.transpose() * S_inv * diff_pred;

            nis[(0, 0)]
        }
    }
    impl SensorMeasurement<RadarMeasurement> for RadarMeasurement {}

    #[derive(Debug)]
    pub struct RadarSensor {
        pub n_z: usize,      // no of measurements
        pub std_radr: f64,   // noise standard deviation radius in m
        pub std_radphi: f64, // noise standard deviaion angle in rad
        pub std_radrd: f64,  // noise standard deviation radius change in m/s
    }

    impl DeviceSensor<RadarSensor> for RadarSensor {
        fn new() -> RadarSensor {
            RadarSensor {
                n_z: N_Z_RADAR,
                std_radr: 0.3,
                std_radphi: 0.03,
                std_radrd: 0.3,
            }
        }
    }

    impl HasSensorNoiseCovar<RadarSensor, RadarNoiseCovarMatrix> for RadarSensor {
        fn noise_covar_matrix(&self) -> RadarNoiseCovarMatrix {
            RadarNoiseCovarMatrix::new(
                self.std_radr * self.std_radr,
                0.0,
                0.0,
                0.0,
                self.std_radphi * self.std_radphi,
                0.0,
                0.0,
                0.0,
                self.std_radrd * self.std_radrd,
            )
        }
    }

    #[derive(Debug)]
    pub enum SensorType {
        Lidar,
        Radar,
    }

    const LIDAR_CHAR: char = 'L';
    const RADAR_CHAR: char = 'R';

    #[derive(Debug)]
    pub struct GroudTruthPackage {
        pub x: f64,
        pub y: f64,
        pub vx: f64,
        pub vy: f64,
    }

    impl GroudTruthPackage {
        pub fn from_csv_string(line: std::string::String) -> GroudTruthPackage {
            let data: Vec<&str> = line.split('\t').collect();

            let n = data.len();
            let data = data[n - 4..n].to_vec();
            let x = data[0].parse::<f64>().unwrap();
            let y = data[1].parse::<f64>().unwrap();
            let vx = data[2].parse::<f64>().unwrap();
            let vy = data[3].parse::<f64>().unwrap();

            GroudTruthPackage {
                x: x,
                y: y,
                vx: vx,
                vy: vy,
            }
        }
    }

    #[derive(Debug)]
    pub struct EstimationPackage {
        pub x: f64,
        pub y: f64,
        pub vx: f64,
        pub vy: f64,
    }

    impl EstimationPackage {
        pub fn from_state(x: &StateVector) -> EstimationPackage {
            EstimationPackage {
                x: x[0],
                y: x[1],
                vx: x[2] * x[3].cos(),
                vy: x[2] * x[3].sin(),
            }
        }

        pub fn residual_vector(&self, gt: &GroudTruthPackage) -> Vector4<f64> {
            Vector4::new(
                self.x - gt.x,
                self.y - gt.y,
                self.vx - gt.vx,
                self.vy - gt.vy,
            )
        }
    }

    #[derive(Debug)]
    pub struct MeasurementPackage {
        pub sensor_type: SensorType,
        pub lidar_data: Option<LidarMeasurement>,
        pub radar_data: Option<RadarMeasurement>,
        raw_measurements: Vec<f64>,
        pub timestamp: u64,
    }

    impl MeasurementPackage {
        pub fn from_csv_string(line: std::string::String) -> MeasurementPackage {
            let data: Vec<&str> = line.split('\t').collect();

            // first value will be sensor type
            return match data[0].chars().next() {
                Some(LIDAR_CHAR) => MeasurementPackage::new_lidar_data(data),
                Some(RADAR_CHAR) => MeasurementPackage::new_radar_data(data),
                Some(_) => panic!("unknown sensor type"),
                None => panic!("no sensor type in data"),
            };
        }

        fn new_lidar_data(data: Vec<&str>) -> MeasurementPackage {
            let px = data[1].parse::<f64>().unwrap();
            let py = data[2].parse::<f64>().unwrap();
            let timestamp = data[3].parse::<u64>().unwrap();

            let lidar_data = Some(LidarMeasurement { px: px, py: py });

            MeasurementPackage {
                sensor_type: SensorType::Lidar,
                lidar_data: lidar_data,
                radar_data: None,
                raw_measurements: vec![px, py],
                timestamp: timestamp,
            }
        }

        fn new_radar_data(data: Vec<&str>) -> MeasurementPackage {
            let rho = data[1].parse::<f64>().unwrap();
            let theta = data[2].parse::<f64>().unwrap();
            let rho_dot = data[3].parse::<f64>().unwrap();
            let timestamp = data[4].parse::<u64>().unwrap();

            let radar_data = Some(RadarMeasurement {
                rho: rho,
                theta: theta,
                rho_dot: rho_dot,
            });

            MeasurementPackage {
                sensor_type: SensorType::Radar,
                lidar_data: None,
                radar_data: radar_data,
                raw_measurements: vec![rho, theta, rho_dot],
                timestamp: timestamp,
            }
        }
    }
}
