#!/usr/bin/env bash
set -euo pipefail

mkdir -p models/wav2lip
mkdir -p models/insightface/models

# 1. Wav2Lip-GAN checkpoint (~436 MB).
W2L_URLS=(
  "https://github.com/justinjohn0306/Wav2Lip/releases/download/models/wav2lip_gan.pth"
)
if [ ! -f models/wav2lip/wav2lip_gan.pth ]; then
  for url in "${W2L_URLS[@]}"; do
    echo "Fetching $url"
    if curl -L --fail --retry 3 --retry-delay 2 \
        -o models/wav2lip/wav2lip_gan.pth "$url"; then
      break
    fi
    rm -f models/wav2lip/wav2lip_gan.pth
  done
fi
test -f models/wav2lip/wav2lip_gan.pth || { echo "all wav2lip mirrors failed"; exit 1; }
test "$(stat -c%s models/wav2lip/wav2lip_gan.pth)" -gt 400000000

# 2. Insightface buffalo_l (same detector the scoring uses, so bboxes match exactly).
#    INSIGHTFACE_HOME env is unreliable on this version — use root= constructor arg.
python - <<'PY'
import pathlib
import insightface
root = str(pathlib.Path.cwd() / "models" / "insightface")
app = insightface.app.FaceAnalysis(name="buffalo_l", root=root)
app.prepare(ctx_id=-1, det_size=(640, 640))
print(f"buffalo_l cached under: {root}/models/buffalo_l/")
PY

ls -lah models/wav2lip/wav2lip_gan.pth
ls -la models/insightface/models/buffalo_l/
echo "Weights ready."
