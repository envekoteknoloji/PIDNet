# ----------------------------
# Base image
# ----------------------------
FROM nvcr.io/nvidia/pytorch:25.06-py3

# ----------------------------
# Set environment variables for X11 display
# ----------------------------
ENV DISPLAY=:1
ENV QT_X11_NO_MITSHM=1
ENV NVIDIA_DRIVER_CAPABILITIES=all 
ENV DEBIAN_FRONTEND=noninteractive 

# ----------------------------
# Install apt packages
# ----------------------------
RUN apt-get update 
RUN apt-get install -y x11-apps
RUN apt-get install -y nano 
RUN apt-get install -y htop 
RUN apt-get install -y wget
RUN apt-get install -y curl
RUN apt-get install -y xdg-utils
RUN apt-get install -y libnspr4
RUN apt-get install -y libnss3
RUN apt-get install -y libgl1-mesa-dev

# ----------------------------
# Git safety configuration
# ----------------------------
RUN git config --global --add safe.directory '*'

# ----------------------------
# Install CUDA, CUDNN
# ----------------------------
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
RUN dpkg -i cuda-keyring_1.1-1_all.deb
RUN apt-get update
RUN apt-get install -y cuda-toolkit-12-8
RUN apt-get install -y cudnn
RUN pip install onnxruntime-gpu

# ----------------------------
# Install VS Code
# ----------------------------
RUN wget https://vscode.download.prss.microsoft.com/dbazure/download/stable/2901c5ac6db8a986a5666c3af51ff804d05af0d4/code_1.101.2-1750797935_amd64.deb
RUN dpkg -i code_1.101.2-1750797935_amd64.deb
RUN mkdir -p /home/user/vscode
RUN code --install-extension usernamehw.errorlens --no-sandbox --user-data-dir /home/user/vscode
RUN code --install-extension ms-python.python --no-sandbox --user-data-dir /home/user/vscode
RUN code --install-extension ms-vscode.cpptools --no-sandbox --user-data-dir /home/user/vscode
RUN code --install-extension twxs.cmake --no-sandbox --user-data-dir /home/user/vscode
RUN code --install-extension ms-vscode.cmake-tools --no-sandbox --user-data-dir /home/user/vscode
RUN code --install-extension github.vscode-pull-request-github --no-sandbox --user-data-dir /home/user/vscode

# ----------------------------
# Install pip packages
# ----------------------------
RUN pip install tensorboardX
RUN pip install yacs
RUN pip install numpy==1.26.4
RUN pip install opencv-python
RUN pip install PySide6

# ----------------------------
# Default container command
# ----------------------------
CMD bash -c "cd /home/user/workspace/PIDNet && exec bash"