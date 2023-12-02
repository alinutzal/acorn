# base image
FROM nvcr.io/nvdlfwea/pyg/pyg:23.10-py3

# local and envs
#ENV LANG C.UTF-8
#ENV LC_ALL C.UTF-8
#ENV PIP_ROOT_USER_ACTION=ignore
#ARG DEBIAN_FRONTEND=noninteractive

# add some packages
RUN apt-get update
RUN apt-get install -y git h5utils wget vim g++

# update python pip
#RUN python -m pip install --upgrade pip
#RUN python --version
#RUN python -m pip --version

ENV PYTHONNOUSERSITE=True
COPY . .
#RUN cd /tmp/ && git clone --recursive https://github.com/asnaylor/FRNN.git 
#RUN cd /tmp/FRNN/external/ && git clone https://github.com/lxxue/prefix_sum 
RUN cd FRNN/external/prefix_sum/ \
    && NVCC_FLAGS="-gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86" \
		python setup.py install 
RUN cd FRNN/ \ 
	&& NVCC_FLAGS="-gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86" \
		python setup.py install 
#RUN	rm -rf /tmp/prefix_sum && rm -rf /tmp/FRNN 

RUN pip install atlasify \
    && pip install https://github.com/LAL/trackml-library/tarball/master#egg=trackml \
    && pip install seaborn \
    && pip install lightning \
    && pip install uproot \
    && pip install class-resolver \
    && pip install wandb

# copy and install package
RUN pip install -e .

