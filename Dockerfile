# syntax=docker/dockerfile:1
# --- build stage: modern NVML headers (CUDA 12) to compile the BAR0 temp reader ---
FROM nvidia/cuda:12.6.2-devel-ubuntu22.04 AS build
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends gcc libpci-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY gputemps.c .
# gputemps.c (ThomasBaruzier, Apache-2.0) reads GDDR6/6X junction+VRAM temps off PCIe BAR0
RUN gcc gputemps.c -o gputemps -O3 -I/usr/local/cuda/include \
    -L/usr/local/cuda/lib64/stubs -lnvidia-ml -lpci

# --- runtime stage: slim; the host's real libnvidia-ml is injected by --runtime=nvidia ---
FROM ubuntu:22.04 AS runtime
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      libpci3 python3 python3-paho-mqtt \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /app/gputemps /usr/local/bin/gputemps
COPY exporter.py /usr/local/bin/exporter.py
EXPOSE 9835
ENTRYPOINT ["python3", "/usr/local/bin/exporter.py"]
