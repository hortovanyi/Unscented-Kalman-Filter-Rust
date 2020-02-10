mod sensor;
mod filter;

extern crate clap;

#[macro_use]
extern crate slog;
extern crate slog_term;
extern crate slog_async;
#[macro_use]
extern crate slog_scope;

use slog::{Drain, Logger};
use std::process;
use clap::{Arg, ArgMatches, App, SubCommand, value_t_or_exit};

use std::io;
use std::io::prelude::*;
use std::io::BufReader;
use std::fs::File;
use std::path::Path;
use std::vec::Vec;

pub use sensor::measurement::*;
pub use filter::kalman_filter::*;

fn run_ukf(input_file:File , output_file: File) -> Result<(), String> {
    trace!("run_ukf start");

    let mut measurements: Vec<MeasurementPackage> = Vec::new();
    let mut ground_truths: Vec<GroudTruthPackage> = Vec::new(); 

    let reader = BufReader::new(input_file);

    info!("loading measurement data ....");
    for line in reader.lines(){
        let l = line.unwrap();
        let mp = MeasurementPackage::from_csv_string(l.clone());
        let gtp = GroudTruthPackage::from_csv_string(l.clone());
        // println!("{} {:?} {:?}",l, mp, gtp);
        measurements.push(mp);
        ground_truths.push(gtp);
    }
    
    trace!("creating ukf object");
    let mut ukf:UnscentedKalmanFilter = UKF::new();
    info!("processing measurement data ....");
    for (i, m) in measurements.iter().enumerate() {
        let x = ukf.process_measurement(m);
        // print!("{} {:?} {:?}", i, m, x);
    }
    
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
        error!("input_path: {} does not exist or isn't a file", input_path.display());
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
    run_ukf(input_file, output_file)?;
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
                         <OUTPUT>   'Output file to use'")
                    .arg(Arg::with_name("verbose")
                        .short("v")
                        .multiple(true)
                        .help("verbosity level"))
                    .get_matches();
    
    

    if let Err(err) = run(matches) {
        println!("Application error: {}", err);
        process::exit(1);
    }
}
