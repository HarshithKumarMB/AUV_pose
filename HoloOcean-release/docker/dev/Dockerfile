# NOTICE: This docker image does not contain the Worlds due to licensing restrictions.
# This docker image can be used to develop holoocean client applications, test, and run worlds that are vmounted.
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    libgl1-mesa-glx libglu1-mesa \
    libxrandr2 libxinerama1 libxcursor1 libxi6 \
    libsm6 libxext6 libxrender1 \
    libvulkan1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user 'ue4' with UID 1000 (common best practice)
RUN useradd -m -u 1000 ue4

USER ue4

# Create local user directories owned by ue4
RUN mkdir -p /home/ue4/.local/share

WORKDIR /home/ue4

CMD ["/bin/bash"]