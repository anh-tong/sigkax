FROM nvidia/cuda:11.0-devel-ubuntu18.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install - y git python3-pip

RUN python3 -m pip install -U pip && \
    python3 -m pip install -U "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

COPY entrypoint.sh /entrypoint.sh

ENTRYPOINT [ "/entrypoint.sh" ]