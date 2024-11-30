FROM debian:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    wget \
    curl \
    gnupg \
    build-essential \
    libblas-dev \
    liblapack-dev \
    libcurl4-openssl-dev \
    gfortran \
    && apt-get clean

RUN apt-get update && apt-get install -y python3 python3-pip python3-venv r-base

WORKDIR /app
COPY requirements.txt /app/requirements.txt

RUN python3 -m venv /app/venv \
    && /app/venv/bin/pip install --upgrade pip \
    && if [ -f /app/requirements.txt ]; then /app/venv/bin/pip install -r /app/requirements.txt; fi


ENV VIRTUAL_ENV=/app/venv
ENV PATH="/app/venv/bin:$PATH"

RUN mkdir -p /app/cell
WORKDIR /app/cell

ENTRYPOINT [ "python" ]