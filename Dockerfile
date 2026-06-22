FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    JUPYTER_ENABLE_LAB=yes \
    MPLCONFIGDIR=/workspace/.cache/matplotlib \
    XDG_CACHE_HOME=/workspace/.cache

# Native libraries used by rasterio/GDAL and OpenCV.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        g++ \
        git \
        libgdal-dev \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal \
    C_INCLUDE_PATH=/usr/include/gdal

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r /tmp/requirements.txt

COPY scripts/check_environment.py /usr/local/bin/check_environment.py

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--IdentityProvider.token="]
