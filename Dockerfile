FROM ubuntu:20.04

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-setuptools \
    python3-venv \
    git \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

# install anaconda
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-py39_23.3.1-0-Linux-x86_64.sh \
    && chmod +x Miniconda3-py39_23.3.1-0-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-py39_23.3.1-0-Linux-x86_64.sh -b \
    && rm Miniconda3-py39_23.3.1-0-Linux-x86_64.sh

RUN conda --version

# create a conda env called gp2 with python 3.9
RUN conda create -n gp2 python=3.9 -y

# activate the conda env
RUN echo "source activate gp2" > ~/.bashrc

# pip install gp2
RUN /bin/bash -c "source activate gp2 && pip install gp2"

# finish by openng a bash shell for the user
CMD ["/bin/bash"]