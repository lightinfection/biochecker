FROM nvcr.io/nvidia/l4t-ml:r35.2.1-py3
# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
${NVIDIA_VISIBLE_DEVICES:-all}

ENV NVIDIA_DRIVER_CAPABILITIES \
${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

RUN apt-get update && \
apt-get install -y \
p7zip-full

RUN cd /home/ && git clone https://github.com/lightinfection/cell_recognition.git

