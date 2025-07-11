#!/usr/bin/env python3
import subprocess
import argparse
import os
import csv
import time
import sys

def log(message, verbose=False):
    """Print message only if verbose mode is enabled"""
    if verbose:
        print(message, flush=True)

def export_nsys_data_to_csv(report_file, output_dir=".", verbose=False):
    """Export nsys report data to CSV files for different metrics"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    reports = ["cudaapisum", "gpukernsum", "cudaapitrace"]
    csv_files = {}
    
    for report in reports:
        output_csv = os.path.join(output_dir, f"{os.path.basename(report_file)}")
        csv_out = os.path.join(output_dir, f"{os.path.basename(report_file)}_{report}.csv")
        stats_command = [
            "nsys", "stats", 
            "--format", "csv", 
            "--report", report,
            "--output", output_csv,
            "--force-overwrite", "true",
            report_file
        ]
    
        subprocess.run(stats_command, check=True) #, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        log(f"Exported {report} data to {output_csv}", verbose)
        csv_files[report] = csv_out

    return csv_files

def extract_metrics_from_csv(report_file, output_dir=".", verbose=False):
    """Extract timing metrics from the CSV files generated by nsys"""
    # Export the data to CSV files first
    csv_files = export_nsys_data_to_csv(report_file, output_dir=output_dir, verbose=verbose)

    # Define the base names for the CSV files
    base_name = os.path.basename(report_file)
    gpukernsum_csv = f"{base_name}_gpukernsum.csv"
    cudaapisum_csv = f"{base_name}_cudaapisum.csv"
    
    metrics = {
        'gpu_kernel_total_time': 0,  # Total time spent in GPU kernels
        'cuda_api_total_time': 0,    # Total time spent in CUDA API calls
        'kernel_times': {},          # Time for each kernel
        'api_times': {},             # Time for each API call
        'kernel_count': 0,           # Number of kernel launches
        'memcpy_time': 0,            # Time spent in memory transfers
    }

    # Extract GPU kernel metrics from gpukernsum CSV
    if "gpukernsum" in csv_files and os.path.exists(csv_files["gpukernsum"]):
        try:
            with open(csv_files["gpukernsum"], 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    kernel_name = row['Name'].strip('"')
                    kernel_time = float(row['Total Time (ns)'])
                    metrics['gpu_kernel_total_time'] += kernel_time
                    metrics['kernel_times'][kernel_name] = kernel_time / 1e9  # Convert to seconds
                    metrics['kernel_count'] += int(row['Instances'])
        except Exception as e:
            print(f"Error processing GPU kernel CSV: {e}")
            log(f"Error processing GPU kernel CSV: {e}", verbose)
    
    # Extract CUDA API metrics from cudaapisum CSV
    if "cudaapisum" in csv_files and os.path.exists(csv_files["cudaapisum"]):
        try:
            with open(csv_files["cudaapisum"], 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    api_name = row['Name']
                    api_time = float(row['Total Time (ns)'])
                    metrics['cuda_api_total_time'] += api_time
                    metrics['api_times'][api_name] = api_time / 1e9  # Convert to seconds
                    
                    # Track memory transfer time separately
                    if 'cudaMemcpy' in api_name:
                        metrics['memcpy_time'] += api_time
        except Exception as e:
            print(f"Error processing CUDA API CSV: {e}")
            log(f"Error processing CUDA API CSV: {e}", verbose)
    
    # Convert nanoseconds to seconds for summary metrics
    metrics['gpu_kernel_total_time'] /= 1e9
    metrics['cuda_api_total_time'] /= 1e9
    metrics['memcpy_time'] /= 1e9
    
    return metrics

def run_profiling(args):
    """Run CUDA profiling with the specified arguments"""
    executable_path = args.executable
    
    # Check if executable exists
    if not os.path.exists(executable_path):
        print(f"Error: Executable '{executable_path}' not found.")
        return
    
    print(f"\nStarting CUDA profiling for {os.path.basename(executable_path)}")
    print(f"Testing all combinations of:")
    print(f"  threadsPerBlock = {args.threads_per_block}")
    print(f"  num_blocks = {args.num_blocks}")
    
    if args.extra_args:
        print(f"Extra arguments: {' '.join(args.extra_args)}")
    
    print("\nRunning profiling, please wait...")
    
    results = []
    
    for tpb in args.threads_per_block:
        for nb in args.num_blocks:
            report_filename_base = f"profile_{os.path.basename(executable_path)}_tpb{tpb}_nb{nb}"
            report_filename = f"{report_filename_base}.nsys-rep"
            
            # Build command line arguments
            cmd_args = [str(tpb), str(nb)]
            if args.extra_args:
                cmd_args.extend(args.extra_args)
            
            # Run nsys profile
            profile_command = [
                "nsys", "profile", 
                "--trace=cuda,osrt", "--sample=cpu",
                "--force-overwrite", "true",
                "-o", report_filename,
                executable_path
            ]
            profile_command.extend(cmd_args)            
            print(f"Profiling TPB={tpb}, NB={nb}... ", end="") # , flush=True)
            
            try:
                if args.verbose:
                    subprocess.run(profile_command, check=True)
                else:
                    subprocess.run(profile_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                log("Profile captured.", args.verbose)
                
                # Extract metrics from the generated CSV files
                log(f"Processing report...", args.verbose)
                metrics = extract_metrics_from_csv(report_filename, verbose=args.verbose)
                
                if metrics:
                    # Store the metrics for this configuration
                    results.append({
                        'tpb': tpb, 
                        'nb': nb, 
                        'gpu_kernel_time': metrics['gpu_kernel_total_time'],
                        'cuda_api_time': metrics['cuda_api_total_time'],
                        'memcpy_time': metrics['memcpy_time'],
                        'kernel_times': metrics['kernel_times'],
                        'api_times': metrics['api_times']
                    })
                    
                    print("Done")
                else:
                    print(f"Failed to get metrics")
            except subprocess.CalledProcessError as e:
                print(f"Failed (error code: {e.returncode})")
                if args.verbose:
                    print(f"Error details: {e}")
    
    if not results:
        print("\nNo results collected. Profiling may have failed for all configurations.")
        return
    
    # Print the results in a table format
    print("\n=== PROFILING RESULTS ===")
    print("\nConfiguration Performance Summary:")
    print("-" * 80)
    print(f"{'TPB':<6} | {'NB':<6} | {'Kernel Time (ms)':<16} | {'API Time (ms)':<16} | {'Memcpy Time (ms)':<16}")
    print("-" * 80)
    
    # Sort configurations by GPU kernel time
    sorted_results = sorted(results, key=lambda x: x['gpu_kernel_time'])
    
    for result in sorted_results:
        tpb = result['tpb']
        nb = result['nb']
        kernel_time_ms = result['gpu_kernel_time'] * 1000  # s to ms
        api_time_ms = result['cuda_api_time'] * 1000       # s to ms
        memcpy_time_ms = result['memcpy_time'] * 1000      # s to ms
        
        print(f"{tpb:<6} | {nb:<6} | {kernel_time_ms:<16.3f} | {api_time_ms:<16.3f} | {memcpy_time_ms:<16.3f}")
    
    print("-" * 80)
    
    # Get the best configuration (first in sorted list)
    best_config = sorted_results[0]
    
    # Print detailed breakdown for the best configuration
    print(f"\nBest Configuration: TPB={best_config['tpb']}, NB={best_config['nb']}")
    print(f"GPU Kernel Time: {best_config['gpu_kernel_time']*1000:.3f} ms")
    print(f"CUDA API Time: {best_config['cuda_api_time']*1000:.3f} ms")
    print(f"Memory Copy Time: {best_config['memcpy_time']*1000:.3f} ms")
    
    print("\nKernel Breakdown:")
    print("-" * 80)
    print(f"{'Kernel Name':<50} | {'Time (ms)':<10}")
    print("-" * 80)
    
    # Sort kernels by execution time
    for kernel, time in sorted(best_config['kernel_times'].items(), key=lambda x: x[1], reverse=True):
        time_ms = time * 1000  # s to ms
        print(f"{kernel[:50]:<50} | {time_ms:<10.3f}")
    
    # Clean up temporary files if requested
    if args.clean:
        print("\nCleaning up temporary files...")
        for result in results:
            tpb = result['tpb']
            nb = result['nb']
            report_file = f"profile_{os.path.basename(executable_path)}_tpb{tpb}_nb{nb}.nsys-rep"
            if os.path.exists(report_file):
                os.remove(report_file)
                log(f"Removed {report_file}", args.verbose)
            
            # Also remove CSV files
            for report_type in ["cudaapisum", "gpukernsum", "cudaapitrace"]:
                csv_file = f"{report_file}_{report_type}.csv"
                if os.path.exists(csv_file):
                    os.remove(csv_file)
                    log(f"Removed {csv_file}", args.verbose)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='CUDA Kernel Profiler - Profile CUDA kernels with different thread/block configurations',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('executable', type=str, help='Path to the CUDA executable to profile')
    
    # Optional arguments with defaults
    parser.add_argument('--threads-per-block', '-t', type=int, nargs='+', default=[64, 128, 256, 512, 1024],
                        help='List of threads per block values to test')
    parser.add_argument('--num-blocks', '-b', type=int, nargs='+', default=[32, 64, 128, 256],
                        help='List of number of blocks values to test')
    
    # Other options
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--clean', '-c', action='store_true', help='Clean up temporary files after profiling')
    parser.add_argument('--extra-args', '-e', nargs=argparse.REMAINDER,
                        help='Extra arguments to pass to the executable (must come after all other arguments)')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_profiling(args)
