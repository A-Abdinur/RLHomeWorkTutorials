FROM continuumio/miniconda3:latest
ENV TZ=Europe \
    DEBIAN_FRONTEND=noninteractive

RUN apt update 
RUN apt install -y --no-install-recommends build-essential curl git gnupg2 make cmake ffmpeg swig libz-dev unzip zlib1g-dev
RUN apt install -y --no-install-recommends build-essential lsb-release ack-grep patchelf wget xpra xserver-xorg-dev xvfb python3-pip python3-dev
RUN conda install python=3.8.11  # later versions are not compatible to Open3D 0.13
RUN conda install -c conda-forge glew
RUN conda install -c conda-forge mesalib
RUN conda install -c menpo glfw3

RUN mkdir -p /root/.mujoco \
    && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
    && tar -xf mujoco.tar.gz -C /root/.mujoco \
    && rm mujoco.tar.gz

ENV LD_LIBRARY_PATH /root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

COPY vendor/Xdummy /usr/local/bin/Xdummy
RUN chmod +x /usr/local/bin/Xdummy

# Workaround for https://bugs.launchpad.net/ubuntu/+source/nvidia-graphics-drivers-375/+bug/1674677
COPY ./vendor/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

WORKDIR /mujoco_py
# Copy over just requirements.txt at first. That way, the Docker cache doesn't
# expire until we actually change the requirements.
COPY ./requirements.txt .
COPY ./requirements.dev.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements.dev.txt

# Delay moving in the entire code until the very end.
ENTRYPOINT ["/mujoco_py/vendor/Xdummy-entrypoint"]
CMD ["pytest"]
COPY . /mujoco_py
RUN python setup.py install


WORKDIR /src/
COPY requirements.txt .
RUN pip install -r requirements.txt

# docker build . -t rl_image 
# docker tag rl_image af01/rl_image
# docker push af01/rl_image
# docker run -it --mount type=bind,src="/home/aabdinur/projects/homework_fall2022-main/",target=/src rl_image /bin/bash
