pub mod helper {

    extern crate nalgebra as na;

    use na::Vector4;
    use std::vec::Vec;

    use crate::sensor;

    use sensor::measurement::{EstimationPackage, GroudTruthPackage};

    // normalise radians -pi to pi
    pub fn negative_normalize(radians: f64) -> f64 {
        use std::f64::consts::PI;
        const PI2: f64 = 2.0 * PI;
        let signed_pi = PI.copysign(radians);
        (radians + signed_pi) % PI2 - signed_pi
    }

    pub fn calculate_rmse(
        estimations: &Vec<EstimationPackage>,
        ground_truths: &Vec<GroudTruthPackage>,
    ) -> Vec<f64> {
        let mut rmse = Vector4::new(0.0, 0.0, 0.0, 0.0);

        if estimations.len() != ground_truths.len() || estimations.len() == 0 {
            warn!("Invalid estimation or ground truth data");
            return vec![rmse[0], rmse[1], rmse[2], rmse[3]];
        }

        // accumulate squared residuals
        for i in 0..estimations.len() {
            let mut residual: Vector4<f64> = estimations[i].residual_vector(&ground_truths[i]);

            // coefficient-wise multiplications
            residual.apply(|x| x * x);

            rmse += residual;
        }

        // calculate the mean
        let mut rmse = rmse / estimations.len() as f64;

        // calulate the square root
        rmse.apply(|x| x.sqrt());

        vec![rmse[0], rmse[1], rmse[2], rmse[3]]
    }
}
