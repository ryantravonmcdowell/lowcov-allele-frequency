FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get -o Acquire::Retries=5 install -y --no-install-recommends bcftools tabix less && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --default-timeout=120 --retries=10 \
    numpy==1.26.4 \
    scipy==1.11.4 \
    pandas==2.1.4

WORKDIR /root

COPY data /root/data

RUN mkdir -p /root/results
