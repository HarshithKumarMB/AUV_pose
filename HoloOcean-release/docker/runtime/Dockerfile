# NOTICE: This Docker image can not be shared publicly due to licensing restrictions.
# Any docker image with the Worlds installed can only be shared internally 
# by someone who has accepted the Epic Games EULA.
# Stage 1: Build with pip and install dependencies
FROM frostlab/holoocean:2.0-dev-ubuntu22.04-cuda12.4 AS builder

USER root
# Copy the client folder from the repository. 
# This assumes the build context is set to the root directory of the repository.
COPY ./client /tmp/holoocean_client
RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install /tmp/holoocean_client

# Stage 2: Minimal runtime image
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/venv/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

RUN apt-get update && apt-get install -y \
    python3 \
    libgl1-mesa-glx libglu1-mesa \
    libxrandr2 libxinerama1 libxcursor1 libxi6 \
    libsm6 libxext6 libxrender1 \
    libvulkan1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user 'ue4' with UID 1000
RUN useradd -m -u 1000 ue4

# Copy the virtual environment from the builder
COPY --from=builder /opt/venv /opt/venv

# Install holoocean package content
RUN chown -R ue4:ue4 /opt/venv || true

USER ue4
# Install the worlds
# RUN /opt/venv/bin/python3 -c 'import holoocean; holoocean.install("TestWorlds")'
RUN /opt/venv/bin/python3 -c 'import holoocean; holoocean.install("Ocean")'

WORKDIR /home/ue4

CMD ["/bin/bash"]
