extern crate ndarray;
extern crate rand;

use ndarray::Array;
use ndarray::prelude::*;
use rand::distributions::{Distribution, Uniform};
use rand::thread_rng;

use std::rc::Rc;
use std::cell::RefCell;

pub trait Function {
    fn backward(&self, grad: Rc<RefCell<Array<f32, IxDyn>>>);
}

pub struct Tensor {
    data: Array<f32, IxDyn>,
    grad: Option<Rc<RefCell<Array<f32, IxDyn>>>>,
    requires_grad: bool,
    grad_fn: Option<Box<dyn Function>>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>, requires_grad: bool) -> Rc<RefCell<Self>> {
        let array = Array::from_shape_vec(IxDyn(&shape), data).unwrap();
        Rc::new(RefCell::new(Self {
            data: array,
            grad: None,
            requires_grad,
            grad_fn: None,
        }))
    }

    pub fn zeros(shape: Vec<usize>, requires_grad: bool) -> Rc<RefCell<Self>> {
        let array = Array::<f32, _>::zeros(IxDyn(&shape));
        Rc::new(RefCell::new(Self {
            data: array,
            grad: None,
            requires_grad,
            grad_fn: None,
        }))
    }

    pub fn ones(shape: Vec<usize>, requires_grad: bool) -> Rc<RefCell<Self>> {
        let array = Array::<f32, _>::ones(IxDyn(&shape));
        Rc::new(RefCell::new(Self {
            data: array,
            grad: None,
            requires_grad,
            grad_fn: None,
        }))
    }

    pub fn rand(shape: Vec<usize>, requires_grad: bool) -> Rc<RefCell<Self>> {
        let mut rng = thread_rng();
        let dist = Uniform::new(0.0, 1.0);
        let array = Array::from_shape_fn(IxDyn(&shape), |_| dist.sample(&mut rng));
        Rc::new(RefCell::new(Self {
            data: array,
            grad: None,
            requires_grad,
            grad_fn: None,
        }))
    }

    pub fn add(&self, other: &Tensor) ->  Rc<RefCell<Self>> {
        let result_data = &self.data + &other.data;
        let result_tensor =  Rc::new(RefCell::new(Self {
            data: result_data,
            grad: None,
            requires_grad :self.requires_grad || other.requires_grad ,
            grad_fn: None,
        }));

        if self.requires_grad || other.requires_grad {
            // We use .borrow_mut() to get mutable access to the Tensor inside the Rc<RefCell<>>.
            let mut result_tensor_borrowed = result_tensor.borrow_mut();
            let grad_fn = Box::new(AddBackward {
                operand1: Rc::new(RefCell::new(self.clone())), 
                operand2: Rc::new(RefCell::new(other.clone())), 
            });
            result_tensor_borrowed.grad_fn = Some(grad_fn);
        }
    
        result_tensor
    }


    pub fn sub(&self, other: &Tensor) ->  Rc<RefCell<Self>> {
        let result_data = &self.data - &other.data;
        let result_tensor = Rc::new(RefCell::new(Self {
            data: result_data,
            grad: None,
            requires_grad :self.requires_grad || other.requires_grad ,
            grad_fn: None,
        }));

        if self.requires_grad || other.requires_grad {
            // We use .borrow_mut() to get mutable access to the Tensor inside the Rc<RefCell<>>.
            let mut result_tensor_borrowed = result_tensor.borrow_mut();
            let grad_fn = Box::new(SubBackward {
                operand1: Rc::new(RefCell::new(self.clone())), 
                operand2: Rc::new(RefCell::new(other.clone())), 
            });
            result_tensor_borrowed.grad_fn = Some(grad_fn);
        }
    
        result_tensor
    }

    pub fn mul(&self, other: &Tensor) ->  Rc<RefCell<Self>> {
        let result_data = &self.data * &other.data;
        let result_tensor = Rc::new(RefCell::new(Self {
            data: result_data,
            grad: None,
            requires_grad :self.requires_grad || other.requires_grad ,
            grad_fn: None,
        }));

        if self.requires_grad || other.requires_grad {
            // We use .borrow_mut() to get mutable access to the Tensor inside the Rc<RefCell<>>.
            let mut result_tensor_borrowed = result_tensor.borrow_mut();
            let grad_fn = Box::new(MulBackward {
                operand1: Rc::new(RefCell::new(self.clone())), 
                operand2: Rc::new(RefCell::new(other.clone())), 
            });
            result_tensor_borrowed.grad_fn = Some(grad_fn);
        }
    
        result_tensor
    }

    pub fn div(&self, other: &Tensor) ->  Rc<RefCell<Self>> {
        let result_data = &self.data / &other.data;
        let result_tensor = Rc::new(RefCell::new(Self {
            data: result_data,
            grad: None,
            requires_grad :self.requires_grad || other.requires_grad ,
            grad_fn: None,
        }));

        if self.requires_grad || other.requires_grad {
            // We use .borrow_mut() to get mutable access to the Tensor inside the Rc<RefCell<>>.
            let mut result_tensor_borrowed = result_tensor.borrow_mut();
            let grad_fn = Box::new(DivBackward {
                operand1: Rc::new(RefCell::new(self.clone())), 
                operand2: Rc::new(RefCell::new(other.clone())), 
            });
            result_tensor_borrowed.grad_fn = Some(grad_fn);
        }
    
        result_tensor
    }    
    

    pub fn reshape(&self, new_shape: Vec<usize>, requires_grad: bool) -> Rc<RefCell<Self>> {
        let result = self.data.clone().into_shape(IxDyn(&new_shape)).expect("New shape is incompatible with data size");
        Rc::new(RefCell::new(Self {
            data: result,
            grad: None,
            requires_grad,
            grad_fn: None,
        }))
    }
    

    pub fn shape(&self) -> Vec<usize> {
        self.data.shape().to_vec()
    }

    pub fn data(&self) -> &[f32] {
        self.data.as_slice().expect("Data is not contiguous")
    }    
    
    pub fn backward(&self) {
        if let Some(ref grad_fn) = self.grad_fn {
            let init_grad = Array::ones(self.data.raw_dim());
            let grad = Rc::new(RefCell::new(init_grad));
            grad_fn.backward(grad);
        }
    }

}

struct AddBackward {
    operand1: Rc<RefCell<Tensor>>,
    operand2: Rc<RefCell<Tensor>>,
}

impl Function for AddBackward {
    fn backward(&self, grad: Rc<RefCell<Array<f32, IxDyn>>>) {
        {
            // Scope for operand1 operations
            let mut operand1 = self.operand1.borrow_mut();
            let operand1_grad = operand1.grad.get_or_insert_with(|| Rc::new(RefCell::new(Array::zeros(grad.borrow().raw_dim()))));
            *operand1_grad.borrow_mut() += &*grad.borrow();
        }

        {
            // Scope for operand2 operations
            let mut operand2 = self.operand2.borrow_mut();
            let operand2_grad = operand2.grad.get_or_insert_with(|| Rc::new(RefCell::new(Array::zeros(grad.borrow().raw_dim()))));
            *operand2_grad.borrow_mut() += &*grad.borrow();
        }
    }
}




struct SubBackward {
    operand1: Rc<RefCell<Tensor>>,
    operand2: Rc<RefCell<Tensor>>,
}

impl Function for SubBackward {
    fn backward(&self, grad: Rc<RefCell<Array<f32, IxDyn>>>) {
        {
            // Scope for operand1 operations
            let mut operand1 = self.operand1.borrow_mut();
            let operand1_grad = operand1.grad.get_or_insert_with(|| Rc::new(RefCell::new(Array::zeros(grad.borrow().raw_dim()))));
            *operand1_grad.borrow_mut() += &*grad.borrow();
        }

        {
            // Scope for operand2 operations
            let mut operand2 = self.operand2.borrow_mut();
            let operand2_grad = operand2.grad.get_or_insert_with(|| Rc::new(RefCell::new(Array::zeros(grad.borrow().raw_dim()))));
            *operand2_grad.borrow_mut() -= &*grad.borrow();
        }
    }
}

struct MulBackward {
    operand1: Rc<RefCell<Tensor>>,
    operand2: Rc<RefCell<Tensor>>,
}

impl Function for MulBackward {
    fn backward(&self, grad: Rc<RefCell<Array<f32, IxDyn>>>) {
        let mut operand1_borrowed = self.operand1.borrow_mut();

        if operand1_borrowed.requires_grad {
            let grad_wrt_operand1 = &operand1_borrowed.data * &*grad.borrow(); // B * grad
            let operand1_grad = operand1_borrowed.grad.get_or_insert_with(|| Rc::new(RefCell::new(Array::zeros(grad.borrow().raw_dim()))));
            let mut operand1_grad_borrowed = operand1_grad.borrow_mut();
            *operand1_grad_borrowed += &grad_wrt_operand1;
        }

        let mut operand2_borrowed = self.operand2.borrow_mut();

        if operand2_borrowed.requires_grad {
            let grad_wrt_operand2 = &operand2_borrowed.data * &*grad.borrow(); // A * grad
            let operand2_grad = operand2_borrowed.grad.get_or_insert_with(|| Rc::new(RefCell::new(Array::zeros(grad.borrow().raw_dim()))));
            let mut operand2_grad_borrowed = operand2_grad.borrow_mut();
            *operand2_grad_borrowed += &grad_wrt_operand2;
        }
    }
}

struct DivBackward {
    operand1: Rc<RefCell<Tensor>>,
    operand2: Rc<RefCell<Tensor>>,
}

impl Function for DivBackward {
    fn backward(&self, grad: Rc<RefCell<Array<f32, IxDyn>>>) {
        let mut operand1_borrowed = self.operand1.borrow_mut();

        if operand1_borrowed.requires_grad {            
            let grad_wrt_operand1 = &*grad.borrow() / &operand1_borrowed.data; // grad / B
            let operand1_grad = operand1_borrowed.grad.get_or_insert_with(|| Rc::new(RefCell::new(Array::zeros(grad.borrow().raw_dim()))));
            let mut operand1_grad_borrowed = operand1_grad.borrow_mut();
            *operand1_grad_borrowed += &grad_wrt_operand1;        }

        let mut operand2_borrowed = self.operand2.borrow_mut();


        if operand2_borrowed.requires_grad {            
            let grad_wrt_operand2 = -(&operand1_borrowed.data * &*grad.borrow() / &operand2_borrowed.data.mapv(|x| x.powi(2))); // -A * grad / B^2
            let operand2_grad = operand2_borrowed.grad.get_or_insert_with(|| Rc::new(RefCell::new(Array::zeros(grad.borrow().raw_dim()))));
            let mut operand2_grad_borrowed = operand2_grad.borrow_mut();
            *operand2_grad_borrowed += &grad_wrt_operand2;
        }
    }
}


impl Clone for Tensor {
    fn clone(&self) -> Self {
        Tensor {
            data: self.data.clone(), // Clone the data array
            grad: self.grad.as_ref().map(|g| Rc::new(RefCell::new((*g.borrow()).clone()))), // Clone the gradient if it exists
            requires_grad: self.requires_grad,
            grad_fn: None, // grad_fn is not cloned because it represents a node in the computation graph
        }
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_tensor() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], true);
        let tensor_ref = tensor.borrow();
        assert_eq!(tensor_ref.data(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(tensor_ref.shape(), &[2, 2]);
        assert!(tensor_ref.requires_grad);
    }

    #[test]
    fn test_zeros_tensor() {
        let tensor = Tensor::zeros(vec![2, 2], false);
        let tensor_ref = tensor.borrow();
        assert_eq!(tensor_ref.data(), &[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(tensor_ref.shape(), &[2, 2]);
        assert!(!tensor_ref.requires_grad);
    }

    #[test]
    fn test_ones_tensor() {
        let tensor = Tensor::ones(vec![3, 3], false);
        let tensor_ref = tensor.borrow();
        assert_eq!(tensor_ref.data(), &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        assert_eq!(tensor_ref.shape(), &[3, 3]);
    }

    #[test]
    fn test_rand_tensor() {
        let tensor = Tensor::rand(vec![2, 2], false);
        let tensor_ref = tensor.borrow();
        assert!(tensor_ref.data().iter().all(|&x| x >= 0.0 && x <= 1.0));
        assert_eq!(tensor_ref.shape(), &[2, 2]);
    }

    #[test]
    fn test_add_tensors() {
        let t1 = Tensor::new(vec![1.0, 2.0], vec![2], true);
        let t2 = Tensor::new(vec![3.0, 4.0], vec![2], false);
        let result = t1.borrow().add(&t2.borrow());
        assert_eq!(result.borrow().data().to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_sub_tensors() {
        let t1 = Tensor::new(vec![5.0, 7.0], vec![2], true);
        let t2 = Tensor::new(vec![2.0, 4.0], vec![2], false);
        let result = t1.borrow().sub(&t2.borrow());
        assert_eq!(result.borrow().data().to_vec(), vec![3.0, 3.0]);
    }

    #[test]
    fn test_mul_tensors() {
        let t1 = Tensor::new(vec![1.0, 2.0], vec![2], true);
        let t2 = Tensor::new(vec![3.0, 4.0], vec![2], false);
        let result = t1.borrow().mul(&t2.borrow());
        assert_eq!(result.borrow().data().to_vec(), vec![3.0, 8.0]);
    }

    #[test]
    fn test_div_tensors() {
        let t1 = Tensor::new(vec![8.0, 10.0], vec![2], true);
        let t2 = Tensor::new(vec![2.0, 5.0], vec![2], false);
        let result = t1.borrow().div(&t2.borrow());
        assert_eq!(result.borrow().data().to_vec(), vec![4.0, 2.0]);
    }

    #[test]
    fn test_reshape_tensor() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], true);
        let reshaped = tensor.borrow().reshape(vec![4], true);
        let reshaped_ref = reshaped.borrow();
        assert_eq!(reshaped_ref.shape(), &[4]);
        assert_eq!(reshaped_ref.data(), &[1.0, 2.0, 3.0, 4.0]);
    }
}
