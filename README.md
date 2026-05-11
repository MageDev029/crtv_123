# talkhead-miner

Wav2Lip-GAN based miner for the TalkHead subnet (sn108). Targets the executor's
quality + efficiency rubric directly.

## Build options

The `docker build` step must run somewhere with unrestricted Linux capabilities
(specifically `CAP_SYS_ADMIN` + ability to call `unshare`). The standard
dev container this code was written in **cannot build it locally** — the
container's seccomp filter blocks the `unshare` syscall that every container
runtime needs to spawn a build sandbox. Use one of the paths below instead.

### Option A — GitHub Actions (recommended, no extra infra)

A workflow is included at `.github/workflows/build.yml`. It runs on
GitHub-hosted Ubuntu runners (which have full privileges) and pushes to
Docker Hub.

1. Push this directory to a GitHub repo (the `.gitignore` excludes `models/`
   so the repo stays small — weights are re-downloaded inside the runner).
2. In the repo's **Settings → Secrets and variables → Actions**, add:
   - `DOCKERHUB_USERNAME` — your Docker Hub username
   - `DOCKERHUB_TOKEN` — a Docker Hub access token with write scope
3. Push to `main` (or trigger manually via the Actions tab).
4. Workflow output prints the image digest. Copy it and submit:

   ```bash
   python -m neurons.miner --image-ref "<user>/talkhead-miner@sha256:..."
   ```

Build time: ~15–25 minutes on `ubuntu-latest`.

### Option B — Build on the host that owns this container

If you have shell access to the machine running this dev container, the
host's docker daemon has the privileges this container lacks. Either:

```bash
# from the host, if /bittensor is a bind mount
cd /path/to/my-miner
docker build -t YOURHUB/talkhead-miner:v1 .
docker push YOURHUB/talkhead-miner:v1
docker inspect --format='{{index .RepoDigests 0}}' YOURHUB/talkhead-miner:v1
```

Or copy the directory out first:

```bash
# from the host
docker cp <container-id>:/bittensor/my-miner /tmp/my-miner
cd /tmp/my-miner
docker build -t YOURHUB/talkhead-miner:v1 .
```

### Option C — Re-launch the dev container with privileges

If you control how this container is started, add to the `docker run` invocation:

```
--cap-add SYS_ADMIN --security-opt seccomp=unconfined
```

(or `--privileged` as a sledgehammer). Then `docker build` from inside works.

## Layout

```
my-miner/
├── Dockerfile                # CUDA 11.8 + PyTorch 2.1 + ffmpeg + Wav2Lip
├── docker-entrypoint.sh      # fail-fast check for baked weights
├── download_weights.sh       # fetches wav2lip_gan.pth + InsightFace buffalo_l
├── requirements.txt          # Python deps
├── worker.py                 # /input → /output file-IPC loop
├── wav2lip_runner.py         # the model + scoring-aware post-processing
├── .github/workflows/build.yml
├── .dockerignore
├── .gitignore
└── models/                   # produced by download_weights.sh (gitignored)
```

## Local test under exact executor sandbox flags

Once the image is pushed, validate locally **on a host with a GPU and
docker privileges**:

```bash
mkdir -p /tmp/job/input /tmp/job/output
cp tests/face.png  /tmp/job/input/face.png
cp tests/audio.wav /tmp/job/input/audio.wav
cat > /tmp/job/input/task.json <<'EOF'
{"challenge_id":"t0","text":"","seed":123,"fps":25,"resolution":512,"max_seconds":5}
EOF

docker run -d --rm \
  --gpus all --network=none \
  --cpus=8 --memory=16g --pids-limit=256 \
  --read-only --cap-drop=ALL --security-opt=no-new-privileges \
  --tmpfs /tmp:rw,nosuid,nodev,noexec,size=1g \
  --tmpfs /var/tmp:rw,nosuid,nodev,noexec,size=256m \
  -e TMPDIR=/tmp -e HF_HOME=/tmp/hf \
  -e XDG_CACHE_HOME=/tmp/.cache \
  -v /tmp/job/input:/input:rw \
  -v /tmp/job/output:/output:rw \
  --name w2l-test YOURHUB/talkhead-miner:latest

until [ -f /tmp/job/output/t0.mp4 ]; do sleep 1; done
docker logs w2l-test | tail -20
docker kill w2l-test
```

Then run the executor's exact scoring code against the output (see the
parent repo's `talkhead-executor/`).
