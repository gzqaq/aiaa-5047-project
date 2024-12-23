FROM nvcr.io/nvidia/jax:24.04-py3

RUN apt-get update

RUN python -m pip install -qq \
    transformer_lens==2.9.1 \
    pandas==2.2.3 fastparquet==2024.11.0

RUN python -m pip install -qq scikit-learn==1.6.0
