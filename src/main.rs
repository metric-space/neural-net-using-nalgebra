#[macro_use]
extern crate nalgebra as na;

use na::{
    ComplexField, Const, DefaultAllocator, Dim, DimName, Dynamic, OMatrix, OVector, RealField,
    Scalar,
};
use num::ToPrimitive;
use rand::distributions::{Distribution, Standard};
use rand::seq::SliceRandom;
use rand::thread_rng;

// vector of vector of floats

// nalgebra by default is column vector centric

//trait NN {
//    Output
//    fn forward(self) -> OVector<T,D> ; // gives a vector dependent on the
//    fn backprop(self) -> (OMatrix<T,D,D>, OVector<T,D>); // gives back a tuples of matrices?
//    fn initialize(layers: Vec<T>) -> Self;
//    fn update(&mut self);
//}

struct FeedForward<T: Scalar, D: Dim>
where
    DefaultAllocator: na::allocator::Allocator<T, D, D> + na::allocator::Allocator<T, D>,
{
    layers: Vec<usize>,
    weights: Vec<OMatrix<T, D, D>>,
    biases: Vec<OVector<T, D>>,
}

// fuck traits ans structs for now

fn initialize(layers: Vec<usize>) -> FeedForward<f64, Dynamic>
where
    DefaultAllocator:
        na::allocator::Allocator<f64, Dynamic, Dynamic> + na::allocator::Allocator<f64, Dynamic>,
    Standard: Distribution<f64>,
{
    let sizes = layers.len();

    let mut weights = Vec::new();
    let mut biases = Vec::new();

    for i in 1..sizes {
        weights.push(OMatrix::<f64, Dynamic, Dynamic>::new_random_generic(
            Dynamic::new(layers[i]),
            Dynamic::new(layers[i - 1]),
        ));
        biases.push(OVector::<f64, Dynamic>::new_random_generic(
            Dynamic::new(layers[i]),
            Const::<1>,
        ));
    }

    FeedForward {
        layers: layers,
        weights: weights,
        biases: biases,
    }
}

// sigma
fn sigmoid<T: Scalar + RealField>(x: T) -> T {
    T::zero() / (T::one() + <T as ComplexField>::powf(T::e(), -x))
}
// sigma prime
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

impl<T: Scalar + RealField, D: DimName> FeedForward<T, D>
where
    DefaultAllocator: na::allocator::Allocator<T, D, D> + na::allocator::Allocator<T, D>,
{
    fn forward(&self, input: OVector<T, D>) -> OVector<T, D> {
        self.weights
            .iter()
            .zip(self.biases.iter())
            .fold(input.clone(), |acc, (w, b)| w * acc + b)
    }

    fn backprop(
        &self,
        test_vector: &OVector<T, D>,
        expected_vector: &OVector<T, D>,
    ) -> (Vec<OMatrix<T, D, D>>, Vec<OVector<T, D>>)
    where
        DefaultAllocator: na::allocator::Allocator<T, Const<1>, D>,
    {
        // zeros
        let mut weight_errors = Vec::new();
        let mut bias_errors = Vec::new();

        let l = self.layers.len();

        // this is to be filled up
        // TODO: do I really need this?
        //for i in 1..l {
        //    weight_errors.push(OMatrix::<T, Dynamic, Dynamic>::zeros_generic(Dynamic::new(self.layers[i]), Dynamic::new(self.layers[i-1])));
        //    bias_errors.push(OVector::<T, Dynamic>::zeros_generic(Dynamic::new(self.layers[i]), Const::<1>));
        //}

        for i in 1..l {
            weight_errors.push(OMatrix::<T, D, D>::zeros());
            bias_errors.push(OVector::<T, D>::zeros());
        }

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

        let mut z_collection: Vec<OVector<T, D>> = Vec::new();
        let mut a_collection: Vec<OVector<T, D>> = Vec::new();

        for i in 1..l {
            let z = &self.weights[i] * test_vector + &self.biases[i];
            z_collection.push(z.clone());
            a_collection.push(z.clone().map(|x| sigmoid(x)));
        }

        let mut errors: Vec<OVector<T, D>> = Vec::new();

        for i in (1..l).rev() {
            let err: OVector<T, D>;
            if i == l {
                err = cost_gradient(&expected_vector.clone(), &a_collection[i])
                    .component_mul(&z_collection[i].map(|x| sigmoid_prime(x)));
            } else {
                err = (self.weights[i].transpose() * &errors[i - 1])
                    .component_mul(&z_collection[i - 1].map(|x| sigmoid_prime(x)));
            }

            // TODO: do I really need this?
            errors.push(err.clone());

            weight_errors[i] = err.clone() * (a_collection[i - 1].transpose());
            bias_errors[i] = err.clone();
        }

        (weight_errors, bias_errors)
    }

    fn update(&mut self, batch: Vec<(OVector<T, D>, OVector<T, D>)>, eta: T)
    where
        DefaultAllocator: na::allocator::Allocator<T, Const<1>, D>,
    {
        let mut ww = Vec::new();
        let mut bb = Vec::new();

        let l = self.layers.len();

        // TODO: have to correct this
        for i in 1..l {
            ww.push(OMatrix::<T, D, D>::zeros());
            bb.push(OVector::<T, D>::zeros());
        }

        for (x, y) in &batch {
            let (ws, bs) = self.backprop(x, y);

            ww = ww.iter().zip(ws.iter()).map(|(x, y)| x + y).collect();
            bb = bb.iter().zip(bs.iter()).map(|(x, y)| x + y).collect();
        }

        let bl: T = T::from_usize(batch.len()).unwrap();

        self.weights = self
            .weights
            .iter()
            .zip(ww.iter())
            .map(|(x, y)| x + (y * (-eta.clone() / bl.clone())))
            .collect();
        self.biases = self
            .biases
            .iter()
            .zip(bb.iter())
            .map(|(x, y)| x + (y * (-eta.clone() / bl.clone())))
            .collect();
    }

    fn sgd(
        &mut self,
        training_data: Vec<(OVector<T, D>, OVector<T, D>)>,
        test_data: Vec<(OVector<T, D>, OVector<T, D>)>,
        epochs: usize,
        batch_size: usize,
        eta: T,
    ) where
        DefaultAllocator: na::allocator::Allocator<T, Const<1>, D>,
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

    fn evaluate(&self, test_data: &Vec<(OVector<T, D>, OVector<T, D>)>) -> f32 {
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
    println!("Hello, world!");

    let x: Vec<usize> = Vec::from([700, 50, 10]);
    let a: FeedForward<f64, Dynamic> = initialize(x);
}
