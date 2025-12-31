# Use NVIDIA's official PyTorch container (Optimized for Ada/Hopper)
FROM nvcr.io/nvidia/pytorch:24.10-py3

# Set L40S Architecture (Ada Lovelace)
ENV TORCH_CUDA_ARCH_LIST="8.9"
ENV DEBIAN_FRONTEND=noninteractive

# Install system basics
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 git && rm -rf /var/lib/apt/lists/*

# Fix: Uninstall the container's torch and install the one compatible with Diffusers
RUN pip uninstall -y torch torchvision torchaudio && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install the specific AI libraries
RUN pip install git+https://github.com/huggingface/diffusers && \
    pip install xformers accelerate transformers gradio opencv-python-headless

# Copy your app
WORKDIR /app
COPY . .

CMD ["python3", "app.py"]
