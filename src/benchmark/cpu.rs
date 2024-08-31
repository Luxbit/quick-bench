use std::time::Instant;
use ndarray::linalg::general_mat_mul;
use ndarray::Array2;

pub fn benchmark_cpu(iterations: u64) -> (f64, f64) {
    let size = 1024; // Size of the matrix (nxn) -> produces 2n^3 ops

    // Record the start time
    let start = Instant::now();

    // Pre-allocate matrices
    let mat_a = Array2::<f64>::zeros((size, size));
    let mat_b = Array2::<f64>::zeros((size, size));
    let mut mat_c = Array2::<f64>::zeros((size, size));

    // Perform the matrix multiplication multiple times
    for _ in 0..iterations {
        general_mat_mul(1.0, &mat_a, &mat_b, 0.0, &mut mat_c);
    }

    // Record the end time
    let duration = start.elapsed().as_secs_f64();

    // Calculate the number of floating-point operations
    let flops_per_mul = 2.0 * (size as f64).powi(3);
    let total_flops = flops_per_mul * (iterations as f64);

    // Calculate TFLOPS
    let gflops = total_flops / (duration * 1e9);

    (gflops, duration)
}

