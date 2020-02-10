pub mod measurement {

  
  #[derive(Debug)]
  pub struct LidarMeasurement {
    pub px: f64, 
    pub py: f64, 
  }

  #[derive(Debug)]
  pub struct RadarMeasurement {
    pub rho: f64, 
    pub theta: f64, 
    pub rho_dot: f64, 
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
      let data = data[n-4 .. n].to_vec();
      let x = data[0].parse::<f64>().unwrap();
      let y = data[1].parse::<f64>().unwrap();
      let vx = data[2].parse::<f64>().unwrap();
      let vy = data[3].parse::<f64>().unwrap();
      
      return GroudTruthPackage{x:x, y:y, vx:vx, vy:vy};
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
    pub fn from_csv_string (line: std::string::String) -> MeasurementPackage {
      
      let data: Vec<&str> = line.split('\t').collect();

      // first value will be sensor type
      return match data[0].chars().next() {
        Some(LIDAR_CHAR) => MeasurementPackage::new_lidar_data(data),
        Some(RADAR_CHAR) => MeasurementPackage::new_radar_data(data),
        Some(_) => panic!("unknown sensor type"),
        None => panic!("no sensor type in data"),
      };

    }

    fn new_lidar_data (data: Vec<&str>) -> MeasurementPackage {

      let px = data[1].parse::<f64>().unwrap();
      let py = data[2].parse::<f64>().unwrap();
      let timestamp = data[3].parse::<u64>().unwrap();

      let lidar_data = Some(LidarMeasurement{px:px,py:py});

      return MeasurementPackage{sensor_type: SensorType::Lidar, 
                                lidar_data: lidar_data,
                                radar_data: None,
                                raw_measurements: vec![px, py],
                                timestamp: timestamp};
    }

    fn new_radar_data (data: Vec<&str>) -> MeasurementPackage {

      let rho = data[1].parse::<f64>().unwrap();
      let theta = data[2].parse::<f64>().unwrap();
      let rho_dot = data[3].parse::<f64>().unwrap();
      let timestamp = data[4].parse::<u64>().unwrap();

      let radar_data = Some(RadarMeasurement{rho:rho, theta:theta, rho_dot:rho_dot});

      return MeasurementPackage{sensor_type: SensorType::Radar, 
                                lidar_data: None,
                                radar_data: radar_data,
                                raw_measurements: vec![rho, theta, rho_dot],
                                timestamp: timestamp};
    }
  }
}