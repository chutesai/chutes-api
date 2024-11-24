FROM nvidia/cuda:12.2.2-devel-ubuntu22.04
RUN apt-get -y update
RUN apt-get -y install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev libexpat1-dev lzma liblzma-dev vim git
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
RUN useradd vllm -s /sbin/nologin
RUN mkdir -p /app /home/vllm && chown vllm:vllm /app /home/vllm
USER vllm
WORKDIR /app
ENV PATH=/opt/python/bin:$PATH
RUN /opt/python/bin/pip install --no-cache 'vllm<0.6.4' wheel packaging
RUN /opt/python/bin/pip install --no-cache flash-attn
RUN /opt/python/bin/pip uninstall -y xformers
# For a pre-built wheel:
# ADD chutes-0.0.27-py3-none-any.whl /tmp/
# RUN /opt/python/bin/pip install /tmp/chutes-0.0.27-py3-none-any.whl

# Or copy from local dir:
ADD --chown=vllm chutes /workspace/chutes
RUN /opt/python/bin/pip install -e chutes

ENV PATH=/home/vllm/.local/bin:$PATH
ADD vllm_example.py /app/vllm_example.py
