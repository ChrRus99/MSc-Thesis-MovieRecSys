import os
import torch
import sys
import subprocess
import logging
from pathlib import Path

def is_docker():
    """Detects if the script is running inside a Docker container."""
    cgroup = Path('/proc/self/cgroup')
    return Path('/.dockerenv').is_file() or cgroup.is_file() and 'docker' in cgroup.read_text()


def main():
    if is_docker():
        log_file = "/opt/airflow/logs/cuda_debug_log.txt"
    else:
        log_file = "D:\\Internship\\recsys\\data_pipelines\\logs\\cuda_debug_log.txt"
    
    logging.basicConfig(level=logging.INFO, filename=log_file, filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("debug_cuda_inner")

    logger.info(f"🚀 Starting CUDA Debugging in {'Docker' if is_docker() else 'Local'} environment...")

    # 1️⃣ Check environment variables
    logger.info("🔍 Checking environment variables...")
    cuda_env_vars = [
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "NVIDIA_DRIVER_CAPABILITIES",
        "TORCH_USE_CUDA_DSA",
        "CUDA_LAUNCH_BLOCKING",
        "LD_LIBRARY_PATH",
        "PATH"
    ]
    for var in cuda_env_vars:
        logger.info(f"{var}: {os.getenv(var)}")

    # 2️⃣ Check nvidia-smi output
    logger.info("🔍 Checking nvidia-smi output...")
    try:
        nvidia_smi = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT).decode()
        logger.info(f"nvidia-smi Output:\n{nvidia_smi}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running nvidia-smi: {e.output.decode()}")

    # 3️⃣ Check nvcc version
    logger.info("🔍 Checking nvcc version...")
    try:
        nvcc_output = subprocess.check_output(["nvcc", "--version"], stderr=subprocess.STDOUT).decode()
        logger.info(f"nvcc Version:\n{nvcc_output}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running nvcc: {e.output.decode()}")

    # 4️⃣ Check PyTorch CUDA availability
    logger.info("🔍 Checking PyTorch CUDA availability...")
    cuda_available = torch.cuda.is_available()
    cuda_version = torch.version.cuda if cuda_available else None
    cuda_device_count = torch.cuda.device_count() if cuda_available else 0
    logger.info(f"CUDA Available: {cuda_available}")
    logger.info(f"CUDA Version: {cuda_version}")
    logger.info(f"CUDA Device Count: {cuda_device_count}")

    if cuda_available and cuda_device_count > 0:
        try:
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            logger.info(f"Current CUDA Device: {current_device}")
            logger.info(f"CUDA Device Name: {device_name}")
        except Exception as e:
            logger.error(f"Error retrieving current device: {e}")

    # 5️⃣ Run ldd on CUDA Libraries
    logger.info("🔍 Checking CUDA library linking (ldd)...")
    cuda_lib_path = "/usr/local/cuda/lib64/libcudart.so"
    try:
        ldd_output = subprocess.check_output(["ldd", cuda_lib_path], stderr=subprocess.STDOUT).decode()
        logger.info(f"CUDA Library Dependencies:\n{ldd_output}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running ldd on {cuda_lib_path}: {e.output.decode()}")

    # 6️⃣ Try Running a Small CUDA Operation
    if cuda_available:
        try:
            logger.info("🔍 Running a small CUDA tensor operation test...")
            test_tensor = torch.tensor([1.0, 2.0, 3.0], device="cuda")
            test_tensor = test_tensor * 2  # Simple GPU computation
            logger.info(f"CUDA Tensor Computation Success: {test_tensor}")
        except Exception as e:
            logger.error(f"Error during CUDA tensor operation: {e}")

    # 7️⃣ Test SentenceTransformer on GPU
    try:
        from sentence_transformers import SentenceTransformer

        logger.info("🔍 Testing SentenceTransformer on GPU...")
        device = "cuda" if cuda_available else "cpu"
        model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        with torch.no_grad():
            sample_texts = ["Hello, world!", "Testing CUDA in Airflow."]
            embeddings = model.encode(sample_texts, convert_to_tensor=True, show_progress_bar=False)
            torch.cuda.synchronize()  # Ensure GPU computations complete
            embeddings = embeddings.cpu()
        logger.info("SentenceTransformer GPU Test: Success ✅")
    except Exception as e:
        logger.error(f"SentenceTransformer CUDA Error: {e}")

    logger.info("✅ CUDA Debugging Completed")

if __name__ == "__main__":
    main()
