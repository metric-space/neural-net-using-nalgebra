extern crate nalgebra as na;

use na::{
    ComplexField, Const, DefaultAllocator, Dim, Dynamic, OMatrix, OVector, RealField, Scalar,
};
use num::ToPrimitive;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rand_distr::StandardNormal;

// ================== IO Utils =======================================================

use std::fs;
use std::ops::AddAssign;

fn mnist_data() -> Vec<(OVector<f64, Dynamic>, OVector<f64, Dynamic>)> {
    let mut result = Vec::new();
    for i in 0..10 {
        let x = fs::read(format!("./data/{}", i)).unwrap();
        let mut z: Vec<f64> = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        z[i] = 1.0;

        for j in x.chunks(28 * 28) {
            let y = j.to_vec();
            let g: Vec<f64> = y.iter().map(|x| *x as f64).collect();

            result.push((
                OVector::<f64, Dynamic>::from_vec(g),
                OVector::<f64, Dynamic>::from_vec(z.clone()),
            ));
        }
    }
    result
}

// ==================================================================================

struct FeedForward<T: Scalar + RealField>
where
    DefaultAllocator:
        na::allocator::Allocator<T, Dynamic, Dynamic> + na::allocator::Allocator<T, Dynamic>,
{
    layers: Vec<usize>,
    weights: Vec<OMatrix<T, Dynamic, Dynamic>>,
    biases: Vec<OVector<T, Dynamic>>,
    weight_zeros: Vec<OMatrix<T, Dynamic, Dynamic>> ,
    bias_zeros: Vec<OVector<T, Dynamic>>
}

fn initialize(layers: Vec<usize>) -> FeedForward<f64>
{
    let sizes = layers.len();

    let mut weights = Vec::new();
    let mut biases = Vec::new();

    for i in 1..sizes {
        weights.push(OMatrix::<f64, Dynamic, Dynamic>::from_distribution_generic(
            Dynamic::new(layers[i]),
            Dynamic::new(layers[i - 1]),
            &StandardNormal,
            &mut thread_rng(),
        ));
        biases.push(OVector::<f64, Dynamic>::from_distribution_generic(
            Dynamic::new(layers[i]),
            Const::<1>,
            &StandardNormal,
            &mut thread_rng(),
        ));
    }

    let mut weight_zeros = Vec::new();
    let mut bias_zeros = Vec::new();

    let l = layers.len();

    for i in 1..l {
        weight_zeros.push(OMatrix::<f64, Dynamic, Dynamic>::zeros_generic(
            Dynamic::new(layers[i]),
            Dynamic::new(layers[i - 1]),
        ));
        bias_zeros.push(OVector::<f64, Dynamic>::zeros_generic(
            Dynamic::new(layers[i]),
            Const::<1>,
        ));
    }

    FeedForward {
        layers,
        weights,
        biases,
        weight_zeros,
        bias_zeros
    }
}

fn sigmoid<T: Scalar + RealField>(x: T) -> T {
    T::one() / (T::one() + <T as ComplexField>::powf(T::e(), -x))
}

fn sigmoid_prime<T: Scalar + RealField>(x: T) -> T {
    let x_ = sigmoid(x);
    x_.clone() * (T::one() - x_)
}

// cost function
fn cost_gradient<T: Scalar + RealField, D: Dim>(
    y: &OVector<T, D>,
    a: &OVector<T, D>,
) -> OVector<T, D>
where
    DefaultAllocator: na::allocator::Allocator<T, D>,
{
    return a - y;
}

impl<T: Scalar + RealField> FeedForward<T>
where
    DefaultAllocator:
        na::allocator::Allocator<T, Dynamic, Dynamic> + na::allocator::Allocator<T, Dynamic>,
{
    fn forward(&self, input: OVector<T, Dynamic>) -> OVector<T, Dynamic> {
        self.weights
            .iter()
            .zip(self.biases.iter())
            .fold(input.clone(), |acc, (w, b)| (w * acc + b).map(sigmoid))
    }

    fn backprop(
        &self,
        test_vector: &OVector<T, Dynamic>,
        expected_vector: &OVector<T, Dynamic>,
    ) -> (Vec<OMatrix<T, Dynamic, Dynamic>>, Vec<OVector<T, Dynamic>>)
    where
        DefaultAllocator: na::allocator::Allocator<T, Const<1>, Dynamic>,
    {
        // zeros
        let mut weight_errors = self.weight_zeros.clone();
        let mut bias_errors = self.bias_zeros.clone();

        // calculate the
        // in reverse
        // go over
        // error = gradient C (*) sigmoidprime(z)
        // error = (W^T * error latest )  (*) sigmoid_prime(z earliest)
        //
        // take a test vector
        //
        // sequentially generate vectors of z
        // sequentially generate vectors of a
        // and then in reverse start calculating errors

        let mut z_collection: Vec<OVector<T, Dynamic>> = Vec::new();
        let mut a_collection: Vec<OVector<T, Dynamic>> = Vec::new();

        a_collection.push(test_vector.clone());

        let l = self.layers.len();

        for i in 0..(l - 1) {
            //println!("{:?}", &self.weights[i].shape());
            //println!("{:?}", test_vector.shape());
            //println!("{:?}", &self.biases[i].shape());
            let z = &self.weights[i] * &a_collection[i] + &self.biases[i];
            z_collection.push(z.clone());
            a_collection.push(z.clone().map(|x| sigmoid(x)));
        }

        // TODO: initialize this elsewhere to prevent build from scratch all the time?
        let mut errors: Vec<OVector<T, Dynamic>> = Vec::new();
        for i in 1..l {
            errors.push(OVector::<T, Dynamic>::zeros_generic(
                Dynamic::new(self.layers[i]),
                Const::<1>,
            ));
        }

        for (e, i) in (0..l - 1).rev().enumerate() {
            let err: OVector<T, Dynamic>;
            if e == 0 {
                err = cost_gradient(&expected_vector.clone(), &a_collection[i + 1])
                    .component_mul(&z_collection[i].map(|x| sigmoid_prime(x)));
            } else {
                err = (self.weights[i + 1].transpose() * &errors[i + 1])
                    .component_mul(&z_collection[i].map(|x| sigmoid_prime(x)));
            }

            errors.push(err.clone());

            weight_errors[i] = err.clone() * (a_collection[i].transpose());
            bias_errors[i] = err.clone();
        }

        (weight_errors, bias_errors)
    }

    fn update(&mut self, batch: Vec<(OVector<T, Dynamic>, OVector<T, Dynamic>)>, eta: T)
    where
        DefaultAllocator: na::allocator::Allocator<T, Const<1>, Dynamic>,
    {
        let mut ww = self.weight_zeros.clone();
        let mut bb = self.bias_zeros.clone();

        for (x, y) in &batch {
            let (ws, bs) = self.backprop(x, y);

            ww.iter_mut().zip(ws.iter()).for_each(|(x, y)| x.add_assign(y));
            bb.iter_mut().zip(bs.iter()).for_each(|(x, y)| x.add_assign(y));
        }

        let bl: T = T::from_usize(batch.len()).unwrap();

         self.weights
            .iter_mut()
            .zip(ww.iter())
            .for_each(|(x, y)| x.add_assign(y * (-eta.clone() / bl.clone())));
        self.biases
            .iter_mut()
            .zip(bb.iter())
            .for_each(|(x, y)| x.add_assign(y * (-eta.clone() / bl.clone())));
    }

    fn sgd(
        &mut self,
        training_data: Vec<(OVector<T, Dynamic>, OVector<T, Dynamic>)>,
        test_data: Vec<(OVector<T, Dynamic>, OVector<T, Dynamic>)>,
        epochs: usize,
        batch_size: usize,
        eta: T,
    ) where
        DefaultAllocator: na::allocator::Allocator<T, Const<1>, Dynamic>,
    {
        for i in 1..epochs {
            let mut training_ = training_data.clone();
            training_.shuffle(&mut thread_rng());

            for batch in training_.chunks(batch_size) {
                self.update(batch.to_vec(), eta.clone());
            }

            let output: f32 = self.evaluate(&test_data);
            println!("{} epoch: evaluation {} ", i, output)
        }
    }

    fn evaluate(&self, test_data: &Vec<(OVector<T, Dynamic>, OVector<T, Dynamic>)>) -> f32 {
        let mut success = 0.0;
        for (x, y) in test_data {
            let a = self.forward(x.clone());
            if a.iamax_full() == y.iamax_full() {
                success += 1.0;
            }
        }

        (success * 100.0) / (test_data.len().to_f32().unwrap())
    }
}

fn main() {
    let x: Vec<usize> = Vec::from([784, 30, 10]);
    let mut f: FeedForward<f64> = initialize(x);
    let mut data = mnist_data();
    let l = data.len();
    data.shuffle(&mut thread_rng());
    let training_l = (0.7 * l.to_f64().unwrap()).to_usize().unwrap();

    let training_data = (&data[0..training_l]).to_vec();
    let testing_data = (&data[training_l..]).to_vec();
    f.sgd(training_data, testing_data, 30, 10, 1.0);
}
