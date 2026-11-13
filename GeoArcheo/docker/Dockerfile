# Dockerfile: ArcheoZS experiments
#
# Contains code + libraries only. Data, outputs and the Prithvi weight cache
# are mounted at run time (see the docker run command in the README section
# or below), so the image stays small and EAMENA data is never baked in.
#
# Build (on the AWS GPU machine, inside the GeoArcheo folder):
#   docker build -t geoarcheo .
#
# Run the site-detection experiment:
#   docker run --gpus all --rm \
#     -v "$PWD/data:/app/data" -v "$PWD/outputs:/app/outputs" -v "$HOME/hf_cache:/hf_cache" \
#     geoarcheo
#
# Run any other step, e.g.:
#   docker run --gpus all --rm -v "$PWD/data:/app/data" -v "$PWD/outputs:/app/outputs" \
#     -v "$HOME/hf_cache:/hf_cache" geoarcheo python -u -m pipeline.background_check

FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/hf_cache

# gdal-bin, make and curl are used by the `elevation` package (SRTM downloads)
RUN apt-get update \
 && apt-get install -y --no-install-recommends gdal-bin make curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-docker.txt .
RUN pip install -r requirements-docker.txt

COPY configs ./configs
COPY models ./models
COPY pipeline ./pipeline
COPY utils ./utils
COPY *.py ./

CMD ["python", "-u", "-m", "pipeline.detect_crossval"]
