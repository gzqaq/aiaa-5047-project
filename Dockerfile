FROM nvcr.io/nvidia/jax:24.04-py3

RUN apt-get update

RUN curl -sSL https://install.python-poetry.org | /usr/bin/python3 -

ARG POETRY_CLI=/root/.local/bin/poetry

COPY . aiaa-5047-project
WORKDIR aiaa-5047-project

RUN /usr/bin/python3 -m pip install .
