# Local NVIDIA GPU setup

Run `./scripts/start-local.sh` from the repository root, then open
<http://localhost:5175/#app>. Keep the terminal open; Ctrl+C stops the services.
The launcher puts service logs in `.local/logs/`.

## What uses the GPU

- Ollama runs the configured `openshorts-local` model (Qwen 2.5 7B, 16K context).
- faster-whisper uses CTranslate2's CUDA backend for word-level transcription.
- FFmpeg uses NVIDIA NVENC for supported video encodes and opportunistic NVDEC
  decoding when enough memory is free.
- TransNetV2 scene inference stays on CPU. The installed PyTorch build is CPU-only.
  This leaves memory for Ollama and Whisper on an 8GB GPU.
- Caption layout and other FFmpeg filters can still use CPU. This is not the
  optional Rust `osrender` GPU compositor.

## Configuration

The local `.env` enables:

```dotenv
WHISPER_MODEL=.cache/whisper-small
WHISPER_DEVICE=cuda
WHISPER_COMPUTE=int8_float16
TRANSCRIBE_BACKEND=whisper
FFMPEG_ENCODER=auto
MAX_CONCURRENT_JOBS=1
ASR_GPU_CONCURRENCY=1
ASR_HOST_SLOTS=1
TRANSNETV2_DEVICE=cpu
```

The actual model path in `.env` may be absolute. The small multilingual transcription model
keeps the initial test small enough to share the card with the 7B LLM. Larger
Whisper models require retesting the combined VRAM usage.

CTranslate2 needs CUDA 12 cuBLAS and cuDNN 9, even if the system's NVIDIA driver
supports a newer CUDA version. The local venv contains these repository-pinned
runtime packages:

```text
nvidia-cublas-cu12==12.9.3.13
nvidia-cudnn-cu12==9.28.0.42
nvidia-cuda-nvrtc-cu12==12.9.86
```

To reinstall those runtimes into the existing venv:

```bash
uv pip install --python .venv/bin/python -r requirements-local-gpu.txt
```

The launcher adds their `nvidia/cublas/lib`, `nvidia/cudnn/lib` and
`nvidia/cuda_nvrtc/lib` directories to
`LD_LIBRARY_PATH` **before** starting Python. The NVIDIA driver remains supplied
by the host. No driver reinstall or system CUDA replacement is needed.

## Checking GPU access

Run `nvidia-smi` in a regular terminal. Restricted execution sandboxes can hide
the GPU and report a driver error even when the host driver works.

An actual NVENC probe:

```bash
ffmpeg -hide_banner -f lavfi -i color=s=256x256:d=1 -c:v h264_nvenc -f null -
```

`FFMPEG_ENCODER=auto` falls back to x264 if the encode probe fails. The existing
Whisper code also retries on CPU after a CUDA runtime error. Inspect processing
logs for either fallback if GPU memory becomes crowded by other applications.

The GUI uses the local backend on port 8000, Vite on 5175 and the editor renderer
on 3100. If these services are already running, stop them before launching again.
