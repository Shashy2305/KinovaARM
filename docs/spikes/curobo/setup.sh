#!/bin/bash
# cuRobo feasibility spike: an ISOLATED environment (micromamba + CUDA 12.4 toolkit + torch 2.4) so the production venv is never touched.
# Everything is logged to setup.log, stage by stage, and runs at the lowest CPU priority because the live stack is busy.
set -u
cd /mnt/ros_workspace/curobo_spike
log() { echo "[$(date +%T)] $*" | tee -a setup.log; }
log "stage 1: micromamba"
if [ ! -x ./bin/micromamba ]; then
  curl -sL https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba || { log "FAILED: micromamba download"; exit 1; }
fi
export MAMBA_ROOT_PREFIX=/mnt/ros_workspace/curobo_spike/mamba
log "stage 2: python 3.10 + CUDA 12.4 toolkit (nvcc) from conda-forge/nvidia"
./bin/micromamba create -y -n cur -c nvidia/label/cuda-12.4.1 -c conda-forge python=3.10 cuda-toolkit=12.4 cuda-nvcc=12.4 git cmake ninja >> setup.log 2>&1 || { log "FAILED: conda toolkit"; exit 2; }
ENV=$MAMBA_ROOT_PREFIX/envs/cur
export CUDA_HOME=$ENV PATH=$ENV/bin:$PATH
log "nvcc: $($ENV/bin/nvcc --version | tail -1)"
log "stage 3: torch 2.4.1 (cu124 wheels)"
$ENV/bin/pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124 >> setup.log 2>&1 || { log "FAILED: torch"; exit 3; }
$ENV/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))" | tee -a setup.log
log "stage 4: clone cuRobo"
[ -d curobo ] || git clone --depth 1 https://github.com/NVlabs/curobo.git >> setup.log 2>&1 || { log "FAILED: clone"; exit 4; }
log "stage 5: build cuRobo CUDA kernels for the RTX 4080 (sm_89); this takes a while"
cd curobo
export TORCH_CUDA_ARCH_LIST="8.9" MAX_JOBS=3
nice -n 19 $ENV/bin/pip install -e . --no-build-isolation >> ../setup.log 2>&1 || { log "FAILED: curobo build (see setup.log)"; exit 5; }
log "stage 6: import test"
$ENV/bin/python -c "import curobo; print('curobo imported from', curobo.__file__)" | tee -a ../setup.log
log "DONE"
