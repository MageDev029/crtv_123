import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "/app")
sys.path.insert(0, "/opt/Wav2Lip")

from wav2lip_runner import Wav2LipRunner


INPUT_DIR = Path(os.environ.get("INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/output"))
TASK_FILE = INPUT_DIR / "task.json"
FACE_FILE = INPUT_DIR / "face.png"
AUDIO_FILE = INPUT_DIR / "audio.wav"
READY_FILE = OUTPUT_DIR / "ready.txt"
POLL_SEC = float(os.environ.get("WORKER_POLL_INTERVAL_SEC", "0.1"))


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    print(f"{ts} {msg}", flush=True)


def main() -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log("MODEL_LOADING_START")
    runner = Wav2LipRunner(
        checkpoint_path="/app/models/wav2lip/wav2lip_gan.pth",
        gpu_id=int(os.environ.get("MUSE_GPU_ID", "0")),
        use_fp16=os.environ.get("USE_FP16", "1") != "0",
        batch_size=int(os.environ.get("BATCH_SIZE", "64")),
    )
    runner.load_model()
    log("MODEL_LOADING_DONE")

    READY_FILE.write_text("ready\n")
    log(f"READY_WRITTEN path={READY_FILE}")

    while True:
        try:
            if not (TASK_FILE.exists() and FACE_FILE.exists() and AUDIO_FILE.exists()):
                time.sleep(POLL_SEC)
                continue

            task = json.loads(TASK_FILE.read_text())
            cid = str(task["challenge_id"])
            log(f"CHALLENGE_RECEIVED challenge_id={cid}")

            start = time.perf_counter()
            output_path = runner.generate(
                face_path=str(FACE_FILE),
                audio_path=str(AUDIO_FILE),
                params={
                    "challenge_id": cid,
                    "fps": task.get("fps", 25),
                    "max_seconds": task.get("max_seconds", 5),
                    "seed": task.get("seed"),
                    "output_dir": str(OUTPUT_DIR),
                },
            )
            elapsed = time.perf_counter() - start

            (OUTPUT_DIR / f"{cid}.json").write_text(json.dumps({
                "challenge_id": cid,
                "success": True,
                "inference_time_sec": elapsed,
                "output_path": output_path,
            }))
            log(f"CHALLENGE_FINISHED challenge_id={cid} time={elapsed:.2f}s")
            TASK_FILE.unlink(missing_ok=True)
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            log(f"ERROR {err}")
            traceback.print_exc()
            cid = "unknown"
            try:
                if TASK_FILE.exists():
                    cid = str(json.loads(TASK_FILE.read_text()).get("challenge_id", "unknown"))
            except Exception:
                pass
            (OUTPUT_DIR / f"{cid}.json").write_text(json.dumps({
                "challenge_id": cid, "success": False, "error": err,
            }))
            TASK_FILE.unlink(missing_ok=True)
            time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
