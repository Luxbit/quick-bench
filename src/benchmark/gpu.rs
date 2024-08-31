use std::time::Instant;
use tch::{Device, Tensor};

pub fn benchmark_gpu(device: Device, iterations: u64) -> (f64, f64) {
    let size = 1024;

    // Create two large tensors filled with random values
    let a = Tensor::rand(&[size, size], (tch::Kind::Float, device));
    let b = Tensor::rand(&[size, size], (tch::Kind::Float, device));

    // Warm up the GPU by performing some operations
    for _ in 0..10 {
        let _ = a.matmul(&b);
    }

    // Record the start time
    let start = Instant::now();

    // Perform the matrix multiplication multiple times
    for _ in 0..iterations {
        let _ = a.matmul(&b);
    }

    // Record the end time
    let duration = start.elapsed().as_secs_f64();

    // Calculate the number of floating-point operations
    let flops_per_mul = 2.0 * (size as f64).powi(3);
    let total_flops = flops_per_mul * (iterations as f64);

    // Calculate TFLOPS
    let tflops = total_flops / (duration * 1e12);

    (tflops, duration)
}