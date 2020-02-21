# Unscented-Kalman-Filter-Rust
UKF written in Rust based on the C++ UKF from the Udacity SD Car Nanodegree. The C++ project code this project is based on can be found at https://github.com/hortovanyi/CarND-Unscented-Kalman-Filter-Project.

To build and run this project use 
```
cargo run -- ./data/sample-laser-radar-measurement-data-1.txt output.txt -vv
``` 
in the project root directory after cloning it.

The above command when run should produce a log with the following output 
```
 Feb 21 09:31:35.370 DEBG slog::Level::TRACE
 Feb 21 09:31:35.371 TRCE app_setup
 Feb 21 09:31:35.371 WARN output_path: output.txt will be overwritten
 Feb 21 09:31:35.371 DEBG opening input for read: `./data/sample-laser-radar-measurement-data-1.txt`
 Feb 21 09:31:35.371 DEBG creating output: `output.txt`
 Feb 21 09:31:35.371 TRCE app_setup_complete
 Feb 21 09:31:35.371 INFO processing_started
 Feb 21 09:31:35.371 TRCE run_ukf start
 Feb 21 09:31:35.372 INFO loading measurement data ....
 Feb 21 09:31:35.395 TRCE creating ukf object
 Feb 21 09:31:35.395 INFO processing measurement data ....
 Feb 21 09:31:35.395 TRCE init_state_radar x:Matrix { data: [8.462918745489562, 0.24346236596519058, -3.04035, 0.0287602, 0.0] }
 Feb 21 09:31:35.395 DEBG init x:
  ┌                     ┐
  │   8.462918745489562 │
  │ 0.24346236596519058 │
  │            -3.04035 │
  │           0.0287602 │
  │                   0 │
  └                     ┘


 Feb 21 09:31:35.395 DEBG init P:
  ┌           ┐
  │ 1 0 0 0 0 │
  │ 0 1 0 0 0 │
  │ 0 0 1 0 0 │
  │ 0 0 0 1 0 │
  │ 0 0 0 0 1 │
  └           ┘


 Feb 21 09:31:35.396 DEBG init lidar_sensor:LidarSensor { n_z: 2, std_laspx: 0.15, std_laspy: 0.15 }
 Feb 21 09:31:35.396 DEBG init radar_sensor:RadarSensor { n_z: 3, std_radr: 0.3, std_radphi: 0.03, std_radrd: 0.3 }
 Feb 21 09:31:36.438 INFO Accruacry - RMSE: [0.07726250037271036, 0.08179721812107782, 0.5892783492841654, 0.5742886905052718]
 Feb 21 09:31:36.438 TRCE run_ukf finish
 Feb 21 09:31:36.438 INFO processing_finished
```

Note the `Accruacry - RMSE: [0.07726250037271036, 0.08179721812107782, 0.5892783492841654, 0.5742886905052718]`

This was my first attempt at a project using Rust. It was a great learning exercise. Compared to the c++ version, dynamic vectors & arrays were not used in favour of taking a more strongly typed position. Extensive use of Rust traits were used. The resultant code has abstracted some calculations down into the traits from the main UKF Fitler implmentation.  
