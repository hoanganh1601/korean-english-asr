"""Download sample audio clips from official Hugging Face datasets:
- thetaone-ai/HiKE (Korean-English code-switching)
- Bingsu/zeroth-korean (Korean clean speech)

Uses soundfile directly on audio bytes to avoid torchcodec dependency.
"""

import io
import os
import sys
import json
from pathlib import Path

# Force UTF-8 stdout for Korean characters on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import soundfile as sf
from datasets import load_dataset, Audio

# Keep all caches inside this project folder
PROJECT_DIR = Path(__file__).parent.resolve()
CACHE_DIR = PROJECT_DIR / ".cache" / "huggingface"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(CACHE_DIR)

OUTPUT_DIR = PROJECT_DIR / "data" / "sample_audios"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download_samples():
    manifest = []
    print("Fetching sample audio from thetaone-ai/HiKE (Korean-English mixed)...")
    try:
        hike_ds = load_dataset(
            "thetaone-ai/HiKE",
            split="test",
            streaming=True,
            cache_dir=str(CACHE_DIR)
        )
        hike_ds = hike_ds.cast_column("audio", Audio(decode=False))

        count = 0
        for item in hike_ds:
            if count >= 3:
                break
            audio_info = item.get("audio")
            if not audio_info:
                continue

            audio_bytes = audio_info.get("bytes")
            if not audio_bytes and audio_info.get("path"):
                with open(audio_info["path"], "rb") as f:
                    audio_bytes = f.read()

            arr, sr = sf.read(io.BytesIO(audio_bytes))
            text = item.get("text", "")
            utterance_id = item.get("utterance_id", f"hike_{count+1}")
            clean_name = f"hike_{count+1}_{utterance_id[:8]}.wav"
            out_path = OUTPUT_DIR / clean_name

            sf.write(str(out_path), arr, sr)
            print(f"  [Saved] {clean_name} ({len(arr)/sr:.2f}s) -> '{text[:60]}...'")

            manifest.append({
                "file": str(out_path.relative_to(PROJECT_DIR)),
                "source": "HiKE (Korean-English Code-Switch)",
                "duration_sec": len(arr)/sr,
                "reference": text
            })
            count += 1
    except Exception as e:
        print(f"  Error fetching thetaone-ai/HiKE: {e}")

    print("\nFetching sample audio from Bingsu/zeroth-korean (Korean clean)...")
    try:
        zeroth_ds = load_dataset(
            "Bingsu/zeroth-korean",
            split="test",
            streaming=True,
            cache_dir=str(CACHE_DIR)
        )
        zeroth_ds = zeroth_ds.cast_column("audio", Audio(decode=False))

        count = 0
        for item in zeroth_ds:
            if count >= 3:
                break
            audio_info = item.get("audio")
            if not audio_info:
                continue

            audio_bytes = audio_info.get("bytes")
            if not audio_bytes and audio_info.get("path"):
                with open(audio_info["path"], "rb") as f:
                    audio_bytes = f.read()

            arr, sr = sf.read(io.BytesIO(audio_bytes))
            text = item.get("text", "")
            clean_name = f"zeroth_{count+1}.wav"
            out_path = OUTPUT_DIR / clean_name

            sf.write(str(out_path), arr, sr)
            print(f"  [Saved] {clean_name} ({len(arr)/sr:.2f}s) -> '{text[:60]}...'")

            manifest.append({
                "file": str(out_path.relative_to(PROJECT_DIR)),
                "source": "Zeroth-Korean (Clean Korean)",
                "duration_sec": len(arr)/sr,
                "reference": text
            })
            count += 1
    except Exception as e:
        print(f"  Error fetching Bingsu/zeroth-korean: {e}")

    manifest_path = OUTPUT_DIR / "sample_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\nSamples manifest saved to: {manifest_path}")


if __name__ == "__main__":
    download_samples()
