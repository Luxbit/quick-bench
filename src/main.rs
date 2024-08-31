mod info;
mod helpers;
mod benchmark;

use info::cpu::get_cpu_info;
use benchmark::{cpu::benchmark_cpu, gpu::benchmark_gpu};
use info::gpu::get_gpu_info;
use tch::Device;
use clap::{Arg, Command};
use serde_json::json;
use std::fs::File;
use std::io::{self, Write};

fn main() -> io::Result<()> {
    let matches = Command::new("System Benchmark")
        .version("1.0")
        .about("Benchmarks CPU and GPU performance")
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .value_name("FORMAT")
                .help("Sets the output format: plain or json")
                .default_value("plain"),
        )
        .arg(
            Arg::new("outputFile")
                .short('f')
                .long("outputFile")
                .value_name("FILE")
                .help("Specifies the file to write the output to")
        )
        .get_matches();

    let output_format = matches.get_one::<String>("output").unwrap();
    let output_file = matches.get_one::<String>("outputFile");

    let cpu_info = get_cpu_info();
    let (cpu_gflops, cpu_elapsed_time) = benchmark_cpu(5);

    let output = if output_format == "json" {
        let mut gpu_results = Vec::new();
        let supports_mps = cpu_info.arch == Some("arm64".to_string()) && cpu_info.os == "macos";
        if supports_mps {
            let (gpu_tflops, gpu_elapsed_time) = benchmark_gpu(Device::Mps, 1000);
            gpu_results.push(json!({
                "device": "MPS",
                "tflops": gpu_tflops,
                "duration": gpu_elapsed_time,
            }));
        } else {
            let gpu_infos = get_gpu_info();
            for (index, info) in gpu_infos.into_iter().enumerate() {
                let (gpu_tflops, gpu_elapsed_time) = benchmark_gpu(Device::Cuda(index), 1000);
                gpu_results.push(json!({
                    "device_id": info.device_id,
                    "device": format!("{:?}", info.device),
                    "name": info.name.unwrap_or_else(|| "Not available".to_string()),
                    "total_memory": info.total_memory.unwrap_or(0),
                    "free_memory": info.free_memory.unwrap_or(0),
                    "used_memory": info.used_memory.unwrap_or(0),
                    "tflops": gpu_tflops,
                    "duration": gpu_elapsed_time,
                }));
            }
        }

        serde_json::to_string_pretty(&json!({
            "cpu_info": {
                "os": cpu_info.os,
                "os_version": cpu_info.os_version.unwrap_or_else(|| "Not available".to_string()),
                "total_memory_mb": cpu_info.total_memory,
                "used_memory_mb": cpu_info.used_memory,
                "total_swap_mb": cpu_info.total_swap,
                "used_swap_mb": cpu_info.used_swap,
                "arch": cpu_info.arch.unwrap_or_else(|| "Not available".to_string()),
                "cpu_count": cpu_info.cpu_count,
                "gflops": cpu_gflops,
                "benchmark_duration_seconds": cpu_elapsed_time
            },
            "gpu_info": gpu_results
        }))?
    } else {
        let mut output = String::new();
        output.push_str("=> CPU Bench:\n");

        output.push_str(&format!("OS          : {:?}\n", cpu_info.os));
        output.push_str(&format!(
            "OS version  : {:?}\n",
            cpu_info
                .os_version
                .unwrap_or_else(|| "Not available".to_string())
        ));
        output.push_str(&format!("Memory Total: {} mb\n", cpu_info.total_memory));
        output.push_str(&format!("Memory Used: {} mb\n", cpu_info.used_memory));
        output.push_str(&format!("Swap Total : {} mb\n", cpu_info.total_swap));
        output.push_str(&format!("Swap Used  : {} mb\n", cpu_info.used_swap));
        output.push_str(&format!(
            "CPU architecture: {:?}\n",
            cpu_info
                .arch
                .clone()
                .unwrap_or_else(|| "Not available".to_string())
        ));
        output.push_str(&format!("CPU count   : {}\n", cpu_info.cpu_count));
        output.push_str(&format!("CPU FLOPS   : {:.2} GFLOPS\n", cpu_gflops));
        output.push_str(&format!(
            "CPU benchmark duration: {:.2} seconds\n\n",
            cpu_elapsed_time
        ));

        output.push_str("=> GPU Bench:\n");
        let supports_mps = cpu_info.arch == Some("arm64".to_string()) && cpu_info.os == "macos";
        if supports_mps {
            let (gpu_tflops, gpu_elapsed_time) = benchmark_gpu(Device::Mps, 1000);
            output.push_str(&format!("GPU: integrated (MPS)\n"));
            output.push_str(&format!("GPU FLOPS: {:.2} TFLOPS\n", gpu_tflops));
            output.push_str(&format!(
                "GPU benchmark duration: {:.2} seconds\n",
                gpu_elapsed_time
            ));
        } else {
            let gpu_infos = get_gpu_info();
            for (index, info) in gpu_infos.into_iter().enumerate() {
                output.push_str(&format!("CUDA Device {} Information:\n", info.device_id));
                output.push_str(&format!("  Device: {:?}\n", info.device));
                output.push_str(&format!(
                    "  Name: {:?}\n",
                    info.name.unwrap_or_else(|| "Not available".to_string())
                ));
                output.push_str(&format!(
                    "  Total Memory: {:?}\n",
                    info.total_memory.unwrap_or(0)
                ));
                output.push_str(&format!("  Free Memory: {:?}\n", info.free_memory.unwrap_or(0)));
                output.push_str(&format!("  Used Memory: {:?}\n", info.used_memory.unwrap_or(0)));

                let (gpu_tflops, gpu_elapsed_time) = benchmark_gpu(Device::Cuda(index), 1000);
                output.push_str(&format!("GPU Estimated FLOPS: {:.2} TFLOPS\n", gpu_tflops));
                output.push_str(&format!(
                    "GPU benchmark duration: {:.2} seconds\n",
                    gpu_elapsed_time
                ));
            }
        }

        output
    };

    if let Some(file_path) = output_file {
        let mut file = File::create(file_path)?;
        file.write_all(output.as_bytes())?;
    } else {
        println!("{}", output);
    }

    Ok(())
}
