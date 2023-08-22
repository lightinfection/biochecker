FROM ubuntu:16.04
# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
${NVIDIA_VISIBLE_DEVICES:-all}

ENV NVIDIA_DRIVER_CAPABILITIES \
${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

RUN apt-get update && \
apt-get install -y \
net-tools \
vim \
git \
ssh \
evince \
texlive-latex-extra
#texlive-latex-base 
#texlive-fonts-recommended \
#texlive-fonts-extra

RUN cd /home/ && git clone https://github.com/HarisIqbal88/PlotNeuralNet.git

