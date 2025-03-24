# # Skip installing requirements.txt as it is already installed in the conda environment
# pip install -r requirements.txt


# Install torch and xformers
pip install torch==2.5.0 xformers==0.0.28.post2 --index-url https://download.pytorch.org/whl/cu121
pip install flash-attn==2.7.4.post1 --no-build-isolation


# Build vllm with existing PyTorch installation (https://docs.vllm.ai/en/latest/getting_started/installation.html#use-an-existing-pytorch-installation)
# We used a fork of vllm that is lingua-compatible  (TODO: remove the fork and instead directly convert checkpoints to vllm-compatible format)
mkdir -p tmp
cd tmp
git clone git@github.com:ryoungj/vllm.git
cd vllm
python use_existing_torch.py
pip install -r requirements-build.txt
TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0+PTX" pip install . --no-build-isolation --verbose

echo "Done installing requirements"