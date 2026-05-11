FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/opt/Wav2Lip \
    INSIGHTFACE_HOME=/app/models/insightface

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3-pip \
        curl git xz-utils libgl1-mesa-glx libglib2.0-0 libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# Static FFmpeg with libx264
ENV FFMPEG_DIR=/opt/ffmpeg
RUN mkdir -p "$FFMPEG_DIR" && cd "$FFMPEG_DIR" \
    && curl -sL "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz" -o ffmpeg.tar.xz \
    && tar -xf ffmpeg.tar.xz && rm ffmpeg.tar.xz \
    && DIR=$(find . -maxdepth 1 -type d -name 'ffmpeg-*' | head -1) \
    && ln -sf "$FFMPEG_DIR/$DIR/bin" "$FFMPEG_DIR/bin"
ENV PATH="$FFMPEG_DIR/bin:$PATH"

# PyTorch CUDA 11.8
RUN python -m pip install --no-cache-dir \
        torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
        --index-url https://download.pytorch.org/whl/cu118

# Python deps
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# Vendor Wav2Lip source (model definition + audio preprocessing)
RUN git clone --depth=1 https://github.com/Rudrabha/Wav2Lip.git /opt/Wav2Lip \
    && rm -rf /opt/Wav2Lip/.git

# Bake model weights (built by download_weights.sh on host)
COPY models ./models

# App
COPY wav2lip_runner.py ./wav2lip_runner.py
COPY worker.py ./worker.py
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["python", "worker.py"]
