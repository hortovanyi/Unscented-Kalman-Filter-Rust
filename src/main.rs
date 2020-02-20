mod filter;
mod sensor;
mod ukf_type;
mod util;

extern crate clap;

#[macro_use]
extern crate slog;
extern crate slog_async;
extern crate slog_term;
#[macro_use]
extern crate slog_scope;

extern crate csv;
#[macro_use]
extern crate serde_derive;

use clap::{value_t_or_exit, App, Arg, ArgMatches, SubCommand};
use slog::{Drain, Logger};
use std::process;

use std::fs::File;
use std::io;
use std::io::prelude::*;
use std::io::{BufReader, LineWriter};
use std::path::Path;
use std::vec::Vec;

pub use filter::kalman_filter::*;
pub use sensor::measurement::*;

#[derive(Serialize)]
struct Output {
    px_est: f64,
    py_est: f64,
    vel_abs_est: f64,
    yaw_angle_est: f64,
    yaw_rate_est: f64,
    px_meas: f64,
    py_meas: f64,
    px_gt: f64,
    py_gt: f64,
    vx_gt: f64,
    vy_gt: f64,
    nis_lidar: f64,
    nis_radar: f64,
}

fn run_ukf(input_file: &File, output_file: &File) -> Result<(), String> {
    trace!("run_ukf start");

    let mut measurements: Vec<MeasurementPackage> = Vec::new();
    let mut ground_truths: Vec<GroudTruthPackage> = Vec::new();
    let mut estimations: Vec<EstimationPackage> = Vec::new();

    let reader = BufReader::new(input_file);
    let mut wtr = csv::WriterBuilder::new()
        .delimiter(b'\t')
        .from_writer(output_file);

    info!("loading measurement data ....");
    for line in reader.lines() {
        let l = line.unwrap();
        let mp = MeasurementPackage::from_csv_string(l.clone());
        let gtp = GroudTruthPackage::from_csv_string(l.clone());
        // trace!("{} {:?} {:?}",l, mp, gtp);
        measurements.push(mp);
        ground_truths.push(gtp);
    }

    trace!("creating ukf object");
    let mut ukf: UnscentedKalmanFilter = UKF::new();
    info!("processing measurement data ....");
    for (i, m) in measurements.iter().enumerate() {
        let x = ukf.process_measurement(m);
        // trace!("{} x:{:?} ", i, x);
        let xest = EstimationPackage::from_state(&x);
        estimations.push(xest);

        let (px_meas, py_meas) = match m.sensor_type {
            SensorType::Lidar => m.lidar_data.as_ref().unwrap().point(),
            SensorType::Radar => m.radar_data.as_ref().unwrap().point(),
        };

        let gtp = &ground_truths[i];
        let output = Output {
            px_est: x[0],
            py_est: x[1],
            vel_abs_est: x[2],
            yaw_angle_est: x[3],
            yaw_rate_est: x[4],
            px_meas: px_meas,
            py_meas: py_meas,
            px_gt: gtp.x,
            py_gt: gtp.y,
            vx_gt: gtp.vx,
            vy_gt: gtp.vy,
            nis_lidar: ukf.nis_lidar,
            nis_radar: ukf.nis_radar,
        };
        wtr.serialize(output).unwrap();
    }

    info!(
        "Accruacry - RMSE: {:?}",
        util::helper::calculate_rmse(&estimations, &ground_truths)
    );
    wtr.flush().unwrap();

    trace!("run_ukf finish");
    Ok(())
}

fn run(matches: ArgMatches) -> Result<(), String> {
    let min_log_level = match matches.occurrences_of("verbose") {
        0 => slog::Level::Info,
        1 => slog::Level::Debug,
        2 | _ => slog::Level::Trace,
    };
    // let decorator = slog_term::PlainDecorator::new(std::io::stdout());
    let decorator = slog_term::TermDecorator::new().build();
    let out_drain = slog_term::CompactFormat::new(decorator).build().fuse();
    let out_drain = slog::LevelFilter(out_drain, min_log_level).fuse();

    // let decorator = slog_term::PlainDecorator::new(std::io::stderr());
    // let err_drain = slog_term::CompactFormat::new(decorator).build().fuse();

    // let drain_pair = slog::Duplicate::new(out_drain, err_drain).fuse();

    let drain = slog_async::Async::new(out_drain).build().fuse();

    let logger = slog::Logger::root(drain, o!("version" => env!("CARGO_PKG_VERSION")));
    let _logger_guard = slog_scope::set_global_logger(logger);

    debug!("slog::Level::{}", min_log_level.as_str());
    trace!("app_setup");
    let input_file_name = value_t_or_exit!(matches.value_of("INPUT"), String);
    let input_path = Path::new(&input_file_name);
    if !input_path.is_file() {
        error!(
            "input_path: {} does not exist or isn't a file",
            input_path.display()
        );
        panic!("no input file!");
    }

    let output_file_name = value_t_or_exit!(matches.value_of("OUTPUT"), String);
    let output_path = Path::new(&output_file_name);
    if output_path.is_file() {
        warn!("output_path: {} will be overwritten", output_path.display());
    }

    debug!("opening input for read: `{}`", input_path.display());
    let input_file = File::open(&input_path).unwrap();

    debug!("creating output: `{}`", output_path.display());
    let output_file = File::create(output_path).unwrap();

    trace!("app_setup_complete");
    // starting processing...
    info!("processing_started");
    run_ukf(&input_file, &output_file)?;
    info!("processing_finished");

    Ok(())
}

fn main() {
    let matches = App::new("ukf")
        .version("0.1.0")
        .author("Nick Hortovanyi")
        .about("Unscented Kalman Filter based on C++ version")
        .args_from_usage(
            "<INPUT>    'Sets the measurement input file to use'
                         <OUTPUT>   'Output file to use'",
        )
        .arg(
            Arg::with_name("verbose")
                .short("v")
                .multiple(true)
                .help("verbosity level"),
        )
        .get_matches();

    if let Err(err) = run(matches) {
        println!("Application error: {}", err);
        process::exit(1);
    }
}
