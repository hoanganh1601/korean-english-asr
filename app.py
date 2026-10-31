"""ASR Evaluation Studio Web UI
Korean-English Code-Switching Speech Recognition & Error Analysis UI
With Instant Real-Text Dataset Lookup & GPU Warmup Acceleration
"""

import asyncio
import io
import json
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# HF cache configuration
PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / ".cache" / "huggingface"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR / "hub")

import jiwer
import soundfile as sf
import torch
from aiohttp import web

# Global runner instance & Manifest Index
runner_instance = None
current_model_mode = "lora_v4"  # "lora_v4", "lora_v3", or "base"
MANIFEST_LOOKUP: Dict[str, Dict[str, str]] = {}


def load_or_build_manifest_lookup() -> Dict[str, Dict[str, str]]:
    """Load precomputed manifest lookup or build from datasets/manifests."""
    global MANIFEST_LOOKUP
    if MANIFEST_LOOKUP:
        return MANIFEST_LOOKUP

    cache_file = PROJECT_ROOT / ".cache" / "manifest_lookup.json"
    if cache_file.exists():
        try:
            print("Loading manifest lookup index from cache...")
            with open(cache_file, "r", encoding="utf-8") as f:
                MANIFEST_LOOKUP = json.load(f)
            print(f"Loaded {len(MANIFEST_LOOKUP):,} indexed audio references.")
            return MANIFEST_LOOKUP
        except Exception as e:
            print("Failed to read cache, rebuilding:", e)

    print("Building manifest lookup index from dataset manifests...")
    mapping = {}

    # 1. sample_manifest.json
    p_sample = PROJECT_ROOT / "data" / "sample_audios" / "sample_manifest.json"
    if p_sample.exists():
        with open(p_sample, "r", encoding="utf-8") as f:
            for item in json.load(f):
                fname = Path(item["file"]).name.lower()
                text = item.get("reference", "").strip()
                if text:
                    entry = {"text": text, "source": item.get("source", "Sample Audios")}
                    mapping[fname] = entry
                    mapping[Path(fname).stem] = entry

    # 2. All jsonl manifests
    manifest_paths = list((PROJECT_ROOT / "artifacts" / "manifests").glob("*.jsonl")) + \
                     list((PROJECT_ROOT / "data" / "datasets").glob("**/metadata.jsonl"))

    for mp in manifest_paths:
        source_name = mp.stem
        try:
            with open(mp, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    fname = row.get("audio_filename")
                    if not fname and "audio_path" in row:
                        fname = Path(row["audio_path"]).name
                    if not fname and "utterance_id" in row:
                        fname = row["utterance_id"] + ".wav"

                    text = row.get("text", "") or row.get("text_normalized", "")
                    if fname and text:
                        text = text.strip()
                        src = row.get("source", source_name)
                        entry = {"text": text, "source": src}
                        fname_lower = Path(fname).name.lower()
                        mapping[fname_lower] = entry
                        mapping[Path(fname_lower).stem] = entry
        except Exception as e:
            print(f"Warning reading {mp.name}: {e}")

    # Save cache
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False)
        print(f"Indexed {len(mapping):,} entries and saved cache to {cache_file.name}")
    except Exception as e:
        print("Error saving manifest cache:", e)

    MANIFEST_LOOKUP = mapping
    return MANIFEST_LOOKUP


def lookup_real_text(filename: str) -> Optional[Dict[str, str]]:
    """Look up ground truth text for an audio file by exact name, stem, or UUID prefix."""
    if not filename:
        return None
    index = load_or_build_manifest_lookup()
    clean_name = Path(filename).name.lower()

    # 0. Strip upload prefix if present
    clean_name = re.sub(r"^upload_\d+_", "", clean_name)

    # 1. Exact match
    if clean_name in index:
        return index[clean_name]

    # 2. Stem match (without extension)
    stem = Path(clean_name).stem
    if stem in index:
        return index[stem]

    # 3. UUID regex extraction match
    uuid_match = re.search(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", clean_name)
    if uuid_match:
        uuid_str = uuid_match.group(1)
        if uuid_str in index:
            return index[uuid_str]
        for ext in (".wav", ".flac", ".mp3"):
            if f"{uuid_str}{ext}" in index:
                return index[f"{uuid_str}{ext}"]
        prefix = uuid_str[:8]
        for k, v in index.items():
            if k.startswith(prefix):
                return v

    # 4. Standard prefix match (first 8 characters)
    prefix = stem[:8]
    if len(prefix) >= 6:
        for k, v in index.items():
            if k.startswith(prefix):
                return v

    return None


from src.evaluation.buzzasr_runner import normalize_for_evaluation


def normalize_text(text: str) -> str:
    """Normalize text for evaluation using the comprehensive Korean-English normalizer."""
    return normalize_for_evaluation(text)


def compute_edit_distance(ref_tokens: List[str], pred_tokens: List[str]) -> int:
    d = [[0] * (len(pred_tokens) + 1) for _ in range(len(ref_tokens) + 1)]
    for i in range(len(ref_tokens) + 1):
        d[i][0] = i
    for j in range(len(pred_tokens) + 1):
        d[0][j] = j
    for i in range(1, len(ref_tokens) + 1):
        for j in range(1, len(pred_tokens) + 1):
            if ref_tokens[i - 1] == pred_tokens[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])
    return d[len(ref_tokens)][len(pred_tokens)]


def compute_cer(ref: str, pred: str) -> float:
    ref_chars = list(re.sub(r"\s+", "", ref))
    pred_chars = list(re.sub(r"\s+", "", pred))
    if not ref_chars:
        return 0.0 if not pred_chars else 1.0
    return compute_edit_distance(ref_chars, pred_chars) / len(ref_chars)


def analyze_errors(ref_raw: str, pred_raw: str) -> Dict[str, Any]:
    """Perform comprehensive word-level alignment & error diagnostics."""
    ref_norm = normalize_text(ref_raw)
    pred_norm = normalize_text(pred_raw)

    ref_words = ref_norm.split()
    pred_words = pred_norm.split()

    if not ref_words and not pred_words:
        return {
            "wer": 0.0,
            "cer": 0.0,
            "accuracy": 100.0,
            "substitutions": [],
            "deletions": [],
            "insertions": [],
            "repetitions": [],
            "visual_alignment": [],
            "ref_word_count": 0,
            "pred_word_count": 0,
            "counts": {"substitutions": 0, "deletions": 0, "insertions": 0, "repetitions": 0, "correct": 0}
        }

    if not ref_words:
        insertions = [{"pred": w, "index": i} for i, w in enumerate(pred_words)]
        return {
            "wer": 100.0,
            "cer": 100.0,
            "accuracy": 0.0,
            "substitutions": [],
            "deletions": [],
            "insertions": insertions,
            "repetitions": [],
            "visual_alignment": [{"type": "insert", "pred": w} for w in pred_words],
            "ref_word_count": 0,
            "pred_word_count": len(pred_words),
            "counts": {"substitutions": 0, "deletions": 0, "insertions": len(insertions), "repetitions": 0, "correct": 0}
        }

    if not pred_words:
        deletions = [{"ref": w, "index": i} for i, w in enumerate(ref_words)]
        return {
            "wer": 100.0,
            "cer": 100.0,
            "accuracy": 0.0,
            "substitutions": [],
            "deletions": deletions,
            "insertions": [],
            "repetitions": [],
            "visual_alignment": [{"type": "delete", "ref": w} for w in ref_words],
            "ref_word_count": len(ref_words),
            "pred_word_count": 0,
            "counts": {"substitutions": 0, "deletions": len(deletions), "insertions": 0, "repetitions": 0, "correct": 0}
        }

    out = jiwer.process_words(ref_norm, pred_norm)
    cer_val = compute_cer(ref_norm, pred_norm)

    substitutions = []
    deletions = []
    insertions = []
    visual_alignment = []

    alignments = out.alignments[0] if out.alignments else []

    for chunk in alignments:
        r_tokens = ref_words[chunk.ref_start_idx : chunk.ref_end_idx]
        h_tokens = pred_words[chunk.hyp_start_idx : chunk.hyp_end_idx]

        if chunk.type == "equal":
            for r_w, h_w in zip(r_tokens, h_tokens):
                visual_alignment.append({"type": "equal", "ref": r_w, "pred": h_w})
        elif chunk.type == "substitute":
            substitutions.append({
                "ref": " ".join(r_tokens),
                "pred": " ".join(h_tokens),
                "ref_range": [chunk.ref_start_idx, chunk.ref_end_idx],
                "pred_range": [chunk.hyp_start_idx, chunk.hyp_end_idx]
            })
            visual_alignment.append({
                "type": "substitute",
                "ref": " ".join(r_tokens),
                "pred": " ".join(h_tokens)
            })
        elif chunk.type == "delete":
            for w in r_tokens:
                deletions.append({"ref": w, "index": chunk.ref_start_idx})
                visual_alignment.append({"type": "delete", "ref": w, "pred": ""})
        elif chunk.type == "insert":
            for w in h_tokens:
                insertions.append({"pred": w, "index": chunk.hyp_start_idx})
                visual_alignment.append({"type": "insert", "ref": "", "pred": w})

    # Detect Repetitions in Prediction
    repetitions = []
    i = 0
    while i < len(pred_words) - 1:
        if pred_words[i] == pred_words[i + 1]:
            repeat_word = pred_words[i]
            count = 1
            start_idx = i
            while i < len(pred_words) - 1 and pred_words[i] == pred_words[i + 1]:
                count += 1
                i += 1
            repetitions.append({
                "word": repeat_word,
                "count": count,
                "start_index": start_idx,
                "type": f"Lặp liên tiếp {count} lần"
            })
        i += 1

    ref_counts = Counter(ref_words)
    pred_counts = Counter(pred_words)
    for word, p_cnt in pred_counts.items():
        r_cnt = ref_counts.get(word, 0)
        if p_cnt > r_cnt and p_cnt >= 2:
            if not any(r["word"] == word for r in repetitions):
                repetitions.append({
                    "word": word,
                    "count": p_cnt,
                    "diff": p_cnt - r_cnt,
                    "type": f"Xuất hiện {p_cnt} lần (trong câu gốc có {r_cnt} lần)"
                })

    wer_pct = round(out.wer * 100, 2)
    cer_pct = round(cer_val * 100, 2)
    acc_pct = max(0.0, round((1.0 - out.wer) * 100, 2))

    return {
        "wer": wer_pct,
        "cer": cer_pct,
        "accuracy": acc_pct,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "repetitions": repetitions,
        "visual_alignment": visual_alignment,
        "ref_word_count": len(ref_words),
        "pred_word_count": len(pred_words),
        "counts": {
            "substitutions": len(substitutions),
            "deletions": len(deletions),
            "insertions": len(insertions),
            "repetitions": len(repetitions),
            "correct": out.hits
        }
    }


def get_asr_runner(mode="lora_v3"):
    global runner_instance, current_model_mode
    if runner_instance is not None and current_model_mode == mode:
        return runner_instance

    from src.evaluation.buzzasr_runner import WhisperASRRunner

    if mode == "lora_v4":
        adapter = "artifacts/checkpoints/buzzasr_lora_v4/final"
        if not (PROJECT_ROOT / adapter).exists():
            adapter = "artifacts/checkpoints/buzzasr_lora_v3/final"
    elif mode == "lora_v3":
        adapter = "artifacts/checkpoints/buzzasr_lora_v3/final"
    else:
        adapter = None

    if adapter:
        abs_adapter = (PROJECT_ROOT / adapter).resolve()
        if abs_adapter.exists():
            adapter = str(abs_adapter)
        else:
            adapter = None

    print(f"\n[ASR Server] Initializing model mode: {mode} (adapter={adapter})...")
    runner_instance = WhisperASRRunner(
        model_id="BuzzASR/korean",
        adapter_path=adapter,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    runner_instance.warmup()
    current_model_mode = mode
    print(f"[ASR Server] Model ready & warmed up on {runner_instance.device.upper()}!\n")
    return runner_instance


# ---------------- API HANDLERS ---------------- #

async def handle_index(request):
    html_file = PROJECT_ROOT / "src" / "ui" / "index.html"
    if html_file.exists():
        return web.FileResponse(html_file)
    return web.Response(text="UI template not found", status=404)


async def handle_get_samples(request):
    manifest_path = PROJECT_ROOT / "data" / "sample_audios" / "sample_manifest.json"
    if not manifest_path.exists():
        return web.json_response([])
    with open(manifest_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    for s in samples:
        fname = Path(s["file"]).name
        s["url"] = f"/api/audio/{fname}"
    return web.json_response(samples)


async def handle_serve_audio(request):
    filename = request.match_info.get("filename", "")
    file_path = PROJECT_ROOT / "data" / "sample_audios" / filename
    if not file_path.exists():
        upload_path = PROJECT_ROOT / "data" / "sample_audios" / "uploads" / filename
        if upload_path.exists():
            file_path = upload_path
        else:
            # Check in raw hike or zeroth datasets
            raw_hike = PROJECT_ROOT / "data" / "raw" / "hike" / "audio" / filename
            if raw_hike.exists():
                file_path = raw_hike
            else:
                return web.Response(text="Audio file not found", status=404)
    return web.FileResponse(file_path)


async def handle_lookup_text(request):
    """Instant lookup of real text by audio filename."""
    filename = request.query.get("filename", "").strip()
    result = lookup_real_text(filename)
    if result:
        return web.json_response({
            "found": True,
            "filename": filename,
            "text": result["text"],
            "source": result.get("source", "Dataset")
        })
    return web.json_response({"found": False, "filename": filename})


async def handle_evaluate(request):
    data = await request.json()
    ref = data.get("reference", "")
    pred = data.get("prediction", "")
    result = analyze_errors(ref, pred)
    return web.json_response(result)


async def handle_transcribe(request):
    reader = await request.multipart()
    audio_bytes = None
    filename = "uploaded_audio.wav"
    ref_text = ""
    model_mode = "lora_v3"

    while True:
        part = await reader.next()
        if part is None:
            break
        if part.name == "audio":
            filename = part.filename or "upload.wav"
            audio_bytes = await part.read()
        elif part.name == "reference":
            ref_text = (await part.read()).decode("utf-8", errors="replace").strip()
        elif part.name == "model_mode":
            model_mode = (await part.read()).decode("utf-8", errors="replace").strip()

    if not audio_bytes:
        return web.json_response({"error": "No audio file provided"}, status=400)

    # Auto-track real text from manifest if not provided by user
    if not ref_text:
        lookup_res = lookup_real_text(filename)
        if lookup_res:
            ref_text = lookup_res["text"]
            print(f"[ASR Server] Auto-matched Real Text for {filename}: {ref_text}")

    # Save to temp audio file
    uploads_dir = PROJECT_ROOT / "data" / "sample_audios" / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    temp_path = uploads_dir / f"upload_{int(time.time())}_{Path(filename).name}"
    with open(temp_path, "wb") as f:
        f.write(audio_bytes)

    # Run Fast Transcription
    loop = asyncio.get_event_loop()
    def transcribe():
        runner = get_asr_runner(mode=model_mode)
        return runner.transcribe_file(str(temp_path))

    trans_result = await loop.run_in_executor(None, transcribe)
    pred_text = trans_result["transcription"]

    # Compute errors if reference is given
    analysis = analyze_errors(ref_text, pred_text) if ref_text else None

    return web.json_response({
        "transcription": pred_text,
        "reference_text": ref_text,
        "inference_time_sec": trans_result["inference_time_sec"],
        "audio_duration_sec": trans_result["audio_duration_sec"],
        "real_time_factor": trans_result["real_time_factor"],
        "peak_vram_mb": trans_result["peak_vram_mb"],
        "audio_url": f"/api/audio/{temp_path.name}",
        "analysis": analysis
    })


def create_app():
    app = web.Application(client_max_size=100 * 1024 * 1024)
    app.router.add_get("/", handle_index)
    app.router.add_get("/api/samples", handle_get_samples)
    app.router.add_get("/api/audio/{filename}", handle_serve_audio)
    app.router.add_get("/api/lookup_text", handle_lookup_text)
    app.router.add_post("/api/transcribe", handle_transcribe)
    app.router.add_post("/api/evaluate", handle_evaluate)

    # Serve static assets
    static_dir = PROJECT_ROOT / "src" / "ui" / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    app.router.add_static("/static/", path=str(static_dir), name="static")
    return app


def find_free_port(start_port: int = 7860, max_tries: int = 20) -> int:
    import socket
    for port in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return start_port


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Korean-English ASR Comparator UI")
    parser.add_argument("--port", type=int, default=7860, help="Port to run server (default: 7860)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host IP (default: 127.0.0.1)")
    parser.add_argument("--no-preload", action="store_true", help="Skip preloading ASR model at startup")
    args = parser.parse_args()

    actual_port = find_free_port(args.port)
    if actual_port != args.port:
        print(f"⚠️ Port {args.port} đang bận, tự động chuyển sang port khả dụng: {actual_port}")

    # Load manifest lookup index into memory
    load_or_build_manifest_lookup()

    # Preload & Warm up ASR model by default so inference is immediate
    if not args.no_preload:
        print("⚡ Preloading & Warming up ASR model on GPU for ultra-fast response...")
        get_asr_runner("lora_v4")

    app = create_app()
    print(f"\n=======================================================")
    print(f"🚀 Korean-English ASR UI is ready at:")
    print(f"👉 http://localhost:{actual_port}")
    print(f"=======================================================\n")
    web.run_app(app, host=args.host, port=actual_port)


if __name__ == "__main__":
    main()
