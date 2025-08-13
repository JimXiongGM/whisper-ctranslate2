import os
import sys
from typing import Optional, Dict, Any
import datetime

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field


def _ensure_src_on_syspath() -> None:
    """Ensure local src/ is importable as a package path."""
    repo_root = os.path.abspath(os.path.dirname(__file__))
    src_dir = os.path.join(repo_root, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


_ensure_src_on_syspath()

# Now safe to import package
from whisper_ctranslate2.transcribe import Transcribe, TranscriptionOptions  # noqa: E402
from whisper_ctranslate2.writers import get_writer  # noqa: E402


class TranscribeRequest(BaseModel):
    file_path: str = Field(..., description="Absolute or relative path to an audio/video file")
    language: Optional[str] = Field(None, description="Language code, e.g. 'zh'. None for auto-detect")


class TranscribeResponse(BaseModel):
    vtt_path: str


app = FastAPI(title="whisper-ctranslate2 server", version="1.0.0")


def _build_default_options(language: Optional[str]) -> TranscriptionOptions:
    """Construct TranscriptionOptions mirroring CLI defaults."""
    # Temperature fallback sequence like CLI (0 -> 1.0 step 0.2)
    temperature_values = tuple([round(x * 0.2, 10) for x in range(0, 6)])

    return TranscriptionOptions(
        beam_size=5,
        best_of=5,
        patience=1.0,
        length_penalty=1.0,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.4,
        condition_on_previous_text=True,
        prompt_reset_on_temperature=0.5,
        temperature=list(temperature_values),
        initial_prompt=None,
        prefix=None,
        hotwords=None,
        suppress_blank=True,
        suppress_tokens=[-1],
        word_timestamps=False,
        print_colors=False,
        prepend_punctuations="\"'“¿([{-",
        append_punctuations="\"'.。,，!！?？:：”)]}、",
        hallucination_silence_threshold=None,
        vad_filter=False,
        vad_threshold=None,
        vad_min_speech_duration_ms=None,
        vad_max_speech_duration_s=None,
        vad_min_silence_duration_ms=None,
        multilingual=False,
    )


LOADED_TRANSCRIBER: Optional[Transcribe] = None


def _create_transcriber(
    *,
    model: str,
    device: str,
    device_index: int,
    compute_type: str,
    threads: int,
    local_files_only: bool,
    batched: bool,
    batch_size: Optional[int],
) -> Transcribe:
    return Transcribe(
        model_path=model,
        device=device,
        device_index=device_index,
        compute_type=compute_type,
        threads=threads,
        cache_directory=None,
        local_files_only=local_files_only,
        batched=batched,
        batch_size=batch_size,
    )


def _ensure_file_and_output(req: TranscribeRequest) -> str:
    input_path = os.path.abspath(req.file_path)
    if not os.path.exists(input_path):
        raise HTTPException(status_code=400, detail=f"file_path does not exist: {input_path}")

    # Force output directory to be the same as input file's directory
    output_dir = os.path.dirname(input_path) or os.getcwd()
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir



def _compute_vtt_output_path(input_path: str, output_dir: str) -> str:
    """
    Compute the output VTT file path, appending a timestamp to the filename for uniqueness.
    """
    base = os.path.splitext(os.path.basename(input_path))[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base}_{timestamp}.vtt"
    return os.path.abspath(os.path.join(output_dir, filename))


def _writer_args() -> Dict[str, Any]:
    # Writer options mirroring CLI defaults
    return {
        "highlight_words": False,
        "max_line_count": None,
        "max_line_width": None,
        "max_words_per_line": None,
        "pretty_json": False,
    }


async def _run_transcription(req: TranscribeRequest) -> str:
    input_path = os.path.abspath(req.file_path)
    output_dir = _ensure_file_and_output(req)

    options = _build_default_options(req.language)
    if LOADED_TRANSCRIBER is None:
        raise HTTPException(status_code=500, detail="Model is not loaded on server")
    transcriber = LOADED_TRANSCRIBER

    def _do_work() -> str:
        result = transcriber.inference(
            audio=input_path,
            task="transcribe",
            language=req.language if req.language else None,
            verbose=True,
            live=False,
            options=options,
        )

        writer = get_writer("vtt", output_dir)
        writer(result, input_path, _writer_args())
        return _compute_vtt_output_path(input_path, output_dir)

    vtt_path = await run_in_threadpool(_do_work)
    return vtt_path


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_endpoint(req: TranscribeRequest) -> TranscribeResponse:
    vtt_path = await _run_transcription(req)
    return TranscribeResponse(vtt_path=vtt_path)


def _maybe_warn_proxy() -> None:
    # If model download is required and you are behind a proxy, configure it first (project-specific).
    # This repository often uses a 'setproxy' helper before downloads.
    # You can export HTTP(S)_PROXY env vars or run setproxy prior to starting the server.
    pass


if __name__ == "__main__":
    _maybe_warn_proxy()
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser("whisper-ctranslate2 server")
    parser.add_argument("--host", default=os.environ.get("WHISPER_SERVER_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("WHISPER_SERVER_PORT", "8000")))

    # Model/device options at server start only (not per-request)
    parser.add_argument("--model", default=os.environ.get("WHISPER_MODEL", "small"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=os.environ.get("WHISPER_DEVICE", "auto"))
    parser.add_argument("--device_index", type=int, default=int(os.environ.get("WHISPER_DEVICE_INDEX", "0")))
    parser.add_argument(
        "--compute_type",
        choices=[
            "default",
            "auto",
            "int8",
            "int8_float16",
            "int8_bfloat16",
            "int8_float32",
            "int16",
            "float16",
            "float32",
            "bfloat16",
        ],
        default=os.environ.get("WHISPER_COMPUTE_TYPE", "auto"),
    )
    parser.add_argument("--threads", type=int, default=int(os.environ.get("WHISPER_THREADS", "0")))
    parser.add_argument("--local_files_only", type=lambda s: s.lower() in ("1", "true", "yes"), default=os.environ.get("WHISPER_LOCAL_ONLY", "true"))
    parser.add_argument("--batched", type=lambda s: s.lower() in ("1", "true", "yes"), default=os.environ.get("WHISPER_BATCHED", "true"))
    parser.add_argument("--batch_size", type=int, default=int(os.environ.get("WHISPER_BATCH_SIZE", "8")))

    args = parser.parse_args()
    # Normalize booleans for cases where defaults come from environment strings
    for _bool_arg in ("local_files_only", "batched"):
        val = getattr(args, _bool_arg)
        setattr(args, _bool_arg, str(val).lower() in ("1", "true", "yes"))

    # Preload model once at startup
    LOADED_TRANSCRIBER = _create_transcriber(
        model=args.model,
        device=args.device,
        device_index=args.device_index,
        compute_type=args.compute_type,
        threads=args.threads,
        local_files_only=args.local_files_only,
        batched=args.batched,
        batch_size=args.batch_size,
    )

    uvicorn.run(app, host=args.host, port=args.port, reload=False, access_log=True)

"""
python start_server.py --port 8101 --model large-v3 --model_dir ./ --local_files_only true --batched true --batch_size 8

curl -X POST http://localhost:8101/transcribe \
-H "Content-Type: application/json" \
-d '{"file_path": "/home/jimx/codes/whisper-ctranslate2/e2e-tests/dosparlants.mp3", "language": "zh"}'



"""
