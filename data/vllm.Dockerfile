FROM nvidia/cuda:12.6.1-cudnn-devel-ubuntu24.04
RUN apt-get update
RUN apt-get -y install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev libexpat1-dev lzma liblzma-dev
WORKDIR /usr/src
RUN wget https://www.python.org/ftp/python/3.12.7/Python-3.12.7.tgz
RUN tar -xzf Python-3.12.7.tgz
WORKDIR /usr/src/Python-3.12.7
RUN ./configure --enable-optimizations --enable-shared --with-system-expat --with-ensurepip=install --prefix=/opt/python
RUN make -j
RUN make altinstall
WORKDIR /root
RUN ln -s /opt/python/bin/pip3.12 /opt/python/bin/pip
RUN ln -s /opt/python/bin/python3.12 /opt/python/bin/python
RUN echo /opt/python/lib >> /etc/ld.so.conf && ldconfig
RUN rm -rf /usr/src/Python*
RUN apt-get -y install google-perftools git
RUN useradd vllm -s /sbin/nologin
RUN mkdir -p /workspace /home/vllm && chown vllm:vllm /workspace /home/vllm
USER vllm
WORKDIR /workspace
ENV PATH=/opt/python/bin:$PATH
RUN /opt/python/bin/pip install --no-cache vllm==0.6.2 wheel packaging
RUN /opt/python/bin/pip install --no-cache flash-attn==2.6.3
RUN /opt/python/bin/pip uninstall -y xformers
RUN /opt/python/bin/pip install chutes==0.0.16
ENV PATH=/home/vllm/.local/bin:$PATH
ENV LD_PRELOAD=/lib/x86_64-linux-gnu/libtcmalloc.so.4
ADD vllm_example.py /workspace/vllm_example.py
