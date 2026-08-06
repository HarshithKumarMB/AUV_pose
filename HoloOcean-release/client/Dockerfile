# This container is used to package and generate worlds automatically
FROM ghcr.io/epicgames/unreal-engine:runtime

USER root
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git libglib2.0-dev software-properties-common

# OpenCV's runtime dependencies (and other dependencies)
RUN apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6

# Install Pyenv and its dependancies
RUN apt-get update && apt-get install -y --no-install-recommends \
    make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev git
ENV PYENV_ROOT="/opt/pyenv"
ENV PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"
RUN curl https://pyenv.run | bash && chmod -R 755 /opt/pyenv

# Install all required Python versions, set to global, rehash to fix permissions
RUN pyenv install 3.9.19 && \
    pyenv install 3.10.16 && \
    pyenv install 3.11.11 && \
    pyenv install 3.12.8 && \
    pyenv install 3.13.1 && \
    pyenv global 3.13.1 3.12.8 3.11.11 3.10.16 3.9.19 && \
    pyenv rehash && \
    chmod -R 777 /opt/pyenv

RUN pip3 install setuptools wheel tox posix_ipc numpy

# Setup user
USER ue4

CMD /bin/bash
