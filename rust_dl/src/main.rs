extern crate ndarray;
extern crate ndarray_rand;
extern crate rand;
extern crate rand_distr;

use ndarray::Array2;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use rand::SeedableRng;
use rand::rngs::StdRng;

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}


fn main() {
    let mut rng = StdRng::from_entropy();

    // Initialize weights randomly
    let weights = Array2::<f64>::random_using((3, 1), Uniform::new(0.0, 1.0), &mut rng);

    // Example: forward pass for one data point
    let inputs = Array2::<f64>::from_shape_vec((1, 3), vec![0.5, 0.3, 0.2]).unwrap();
    let outputs = inputs.dot(&weights).mapv(sigmoid);

    println!("Outputs: {:?}", outputs);
}
