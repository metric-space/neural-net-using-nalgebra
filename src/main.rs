#[macro_use]
extern crate nalgebra as na;

use na::{Dynamic, OMatrix, OVector, Const, Scalar,Dim, DefaultAllocator, RealField, ComplexField, DimName};
use rand::distributions::{Standard, Distribution};

// vector of vector of floats

// nalgebra by default is column vector centric

//trait NN {
//    Output
//    fn forward(self) -> OVector<T,D> ; // gives a vector dependent on the 
//    fn backprop(self) -> (OMatrix<T,D,D>, OVector<T,D>); // gives back a tuples of matrices?
//    fn initialize(layers: Vec<T>) -> Self;
//    fn update(&mut self);
//}

struct FeedForward<T:Scalar, D:Dim>
where
    DefaultAllocator: na::allocator::Allocator<T, D, D> + na::allocator::Allocator<T, D>
{
    layers: Vec<usize>,
    weights: Vec<OMatrix<T, D, D>>,
    biases: Vec<OVector<T, D>>
}

// fuck traits ans structs for now

fn initialize(layers: Vec<usize>) -> FeedForward<f64, Dynamic>
where DefaultAllocator: na::allocator::Allocator<f64, Dynamic, Dynamic> + na::allocator::Allocator<f64, Dynamic>,
      Standard:Distribution<f64>
{
    let sizes = layers.len();

    let mut weights = Vec::new();
    let mut biases = Vec::new();

    for i in 1..sizes {
        weights.push(OMatrix::<f64, Dynamic, Dynamic>::new_random_generic(Dynamic::new(layers[i]), Dynamic::new(layers[i-1])));
        biases.push(OVector::<f64, Dynamic>::new_random_generic(Dynamic::new(layers[i]), Const::<1>));
    }

    FeedForward{layers:layers, weights: weights, biases: biases}

}

// sigma
fn sigmoid<T:Scalar + RealField>(x:T) -> T{
    T::zero()/(T::one() + <T as ComplexField>::powf(T::e(), -x))
}
// sigma prime
fn sigmoid_prime<T:Scalar + RealField>(x:T) -> T{
    let x_ = sigmoid(x);
    x_.clone()*(T::one()-x_)
}

// cost function
fn cost_gradient<T:Scalar + RealField, D:Dim>(y:&OVector<T,D> , a: &OVector<T,D>) -> OVector<T,D>
where DefaultAllocator: na::allocator::Allocator<T, D>
{
    return  a - y
}

impl<T:Scalar + RealField, D:DimName > FeedForward<T,D>
where
    DefaultAllocator: na::allocator::Allocator<T, D, D> + na::allocator::Allocator<T, D>{
    fn forward(self, input: OVector<T, D>) -> OVector<T, D> {
        self.weights
            .iter()
            .zip(self.biases.iter())
            .fold(input.clone(), |acc, (w,b)| {w*acc + b})
    }

    fn backprop(&self, test_vector:OVector<T,D>, expected_vector:OVector<T,D>) -> (Vec<OMatrix<T, D, D>>,Vec<OVector<T, D>>)
    where DefaultAllocator: na::allocator::Allocator<T,Const<1>,D> {
        // zeros
        let mut weight_errors = Vec::new();
        let mut bias_errors= Vec::new();

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

        let mut z_collection: Vec<OVector<T,D>> = Vec::new();
        let mut a_collection: Vec<OVector<T,D>> = Vec::new();

        for i in 1..l {
            let z = &self.weights[i]*&test_vector + &self.biases[i];
            z_collection.push(z.clone());
            a_collection.push(z.clone().map(|x| {sigmoid(x)}));
        }

        let mut errors: Vec<OVector<T,D>> = Vec::new();

        for i in (1..l).rev() {
            let err: OVector<T,D>;
            if i == l {
                err = cost_gradient(&expected_vector.clone(), &a_collection[i]).component_mul(&z_collection[i].map(|x| sigmoid_prime(x)));
            } else {
                err = (self.weights[i].transpose() * &errors[i-1]).component_mul(&z_collection[i-1].map(|x| {sigmoid_prime(x)}));
            }

            // TODO: do I really need this?
            errors.push(err.clone());

            weight_errors[i] = err.clone() * (a_collection[i-1].transpose());
            bias_errors[i] = err.clone();
        }

        (weight_errors, bias_errors)

    }
}

fn main() {
    println!("Hello, world!");

    let x:Vec<usize> = Vec::from([700,50,10]);
    let a:FeedForward<f64, Dynamic> = initialize(x);
}
