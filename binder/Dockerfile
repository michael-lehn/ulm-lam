# The image Binder (mybinder.org) starts for the live notebook: Ubuntu
# 24.04, LLVM/MLIR 23 from apt.llvm.org, Julia, the ulm.lam plugin, the
# notebook environment, and Pluto behind Jupyter's server proxy.
#
#   docker build -f binder/Dockerfile -t ulm-lam .
#   docker run -p 8888:8888 ulm-lam
#   http://localhost:8888/pluto/open?path=/home/jovyan/notebook/ulm-lam.jl
FROM ubuntu:24.04

ARG NB_USER=jovyan
ARG NB_UID=1000
ENV USER=${NB_USER} HOME=/home/${NB_USER} DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg lsb-release software-properties-common wget \
        build-essential git python3 python3-venv \
    && curl -fsSL https://apt.llvm.org/llvm.sh -o /tmp/llvm.sh \
    && bash /tmp/llvm.sh 23 \
    && apt-get install -y --no-install-recommends \
        mlir-23-tools libmlir-23-dev llvm-23-dev clang-format-23 \
    && (test -e /usr/lib/llvm-23/bin/clang-format || ln -s /usr/bin/clang-format-23 /usr/lib/llvm-23/bin/clang-format) \
    && rm -rf /var/lib/apt/lists/* /tmp/llvm.sh

# Julia (the version the notebook's Manifest was resolved with)
ARG JULIA_VERSION=1.13.3
RUN arch=$(uname -m) && \
    case $arch in x86_64) j=x64; a=x86_64 ;; aarch64) j=aarch64; a=aarch64 ;; esac && \
    curl -fsSL https://julialang-s3.julialang.org/bin/linux/$j/${JULIA_VERSION%.*}/julia-${JULIA_VERSION}-linux-$a.tar.gz \
        | tar -xz -C /opt && ln -s /opt/julia-${JULIA_VERSION}/bin/julia /usr/local/bin/julia

# Jupyter (what Binder starts) with Pluto behind its server proxy
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir \
        notebook jupyterlab jupyter-server-proxy
ENV PATH=/opt/venv/bin:$PATH

RUN userdel -r ubuntu 2>/dev/null; useradd -m -s /bin/bash -u ${NB_UID} ${NB_USER}
COPY --chown=${NB_UID} . ${HOME}
COPY --chown=${NB_UID} binder/jupyter_server_config.py ${HOME}/.jupyter/jupyter_server_config.py
USER ${NB_USER}
WORKDIR ${HOME}

ENV LLVM_PREFIX=/usr/lib/llvm-23 \
    ULM_LAM_BACKEND=julia \
    ULM_LAM_NOTEBOOK_ENV=/home/${NB_USER}/notebook \
    JULIA_DEPOT_PATH=/home/${NB_USER}/.julia \
    JULIA_NUM_THREADS=auto

# the plugin, the notebook's packages, and one run of the notebook: Julia's
# compiled code and ulm.lam's libraries are cached in the image
RUN rm -rf build && scripts/build-plugin.sh \
    && julia --project=notebook -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()' \
    && julia --project=notebook notebook/run_headless.jl notebook/ulm-lam.jl > /tmp/run.log 2>&1 \
    ; grep -A15 "ERROR in cell" /tmp/run.log | head -80; ! grep -q "ERROR in cell" /tmp/run.log
