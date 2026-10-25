FROM python:3.11-slim-bookworm

RUN apt-get update && \
    apt-get -o Acquire::Retries=5 install -y --no-install-recommends bash && \
    rm -rf /var/lib/apt/lists/*

# test dependencies are baked here; nothing is installed at trial time
RUN pip install --no-cache-dir --default-timeout=120 --retries=10 \
    pytest==8.4.1 \
    pytest-json-ctrf==0.3.5 \
    numpy==1.26.4 \
    pandas==2.1.4

# parent directory of the declared artifact, and the verifier log directory
RUN mkdir -p /root/results /logs/verifier

COPY . /tests/
