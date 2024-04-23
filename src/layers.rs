use crate::tensor::Tensor;
use std::borrow::Borrow;
use std::rc::Rc;
use std::cell::RefCell;

// Define a struct for the Linear Layer
struct Linear {
    weights: Rc<RefCell<Tensor>>,
    biases: Rc<RefCell<Tensor>>,
    input_features: usize,
    output_features: usize,
}

impl Linear {
    fn new(input_features: usize, output_features: usize) -> Linear {
        // Initialize weights and biases with a single level of Rc<RefCell<>> wrapping
        let weights = Rc::new(RefCell::new(Tensor::rand(vec![output_features, input_features], true)));
        let biases = Rc::new(RefCell::new(Tensor::zeros(vec![output_features], true)));
    
        Linear {
            weights,
            biases,
            input_features,
            output_features,
        }
    }
    

    fn forward(&self, input: &Tensor) -> Tensor {
        let weights_borrowed = self.weights.borrow();
        let biases_borrowed = self.biases.borrow();
    
        // Explicitly specify the type of weights_borrowed and use matmul for matrix multiplication
        let mut output: Tensor = weights_borrowed.mul(input); // Ensure matmul method exists
        output = output.add(&biases_borrowed); // Adding biases
        output
    }
    
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_forward() {
        let input_features = 2;
        let output_features = 3;
        let linear_layer = Linear::new(input_features, output_features);

        // Create an input tensor
        let input = Tensor::new(vec![0.5, -1.5], vec![input_features], false);

        // Perform forward pass
        let output = linear_layer.forward(&input);

        // Just check the shape of the output to make sure it's correct
        assert_eq!(output.shape(), &[output_features]);
        // Normally you'd want to check the values as well, but they depend on random initialization
    }
}
