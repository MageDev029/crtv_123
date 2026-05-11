#!/usr/bin/env bash
set -e
cd /app

required=(
  "/app/models/wav2lip/wav2lip_gan.pth"
  "/app/models/insightface/models/buffalo_l/det_10g.onnx"
)
for f in "${required[@]}"; do
  if [ ! -f "$f" ]; then
    echo "FATAL: missing required model file: $f"
    exit 1
  fi
done

exec "$@"
