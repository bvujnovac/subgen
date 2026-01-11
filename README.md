[![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg)](https://www.paypal.com/donate/?hosted_button_id=SU4QQP6LH5PF6)
<img src="https://raw.githubusercontent.com/McCloudS/subgen/main/icon.png" width="200">

# Subgen - Bazarr Edition

A Whisper ASR (Automatic Speech Recognition) server designed to work as a subtitle provider for [Bazarr](https://www.bazarr.media/). This simplified edition focuses solely on providing high-quality speech-to-text transcription for subtitle generation.

## Features

- Whisper-based transcription using [faster-whisper](https://github.com/guillaumekln/faster-whisper)
- Subtitle timing refinement via [stable-ts](https://github.com/jianfch/stable-ts)
- GPU (CUDA) and CPU support
- Language detection
- Word-level highlighting (optional)
- Compatible with Bazarr's Whisper provider

## Quick Start

### Docker (Recommended)

```bash
docker run -d \
  --name subgen \
  --gpus all \
  -p 9000:9000 \
  -v ./models:/models \
  -e WHISPER_MODEL=medium \
  -e TRANSCRIBE_DEVICE=cuda \
  mccloud/subgen:latest
```

Or use docker-compose:

```yaml
version: '3.8'
services:
  subgen:
    image: mccloud/subgen:latest
    container_name: subgen
    environment:
      - WHISPER_MODEL=medium
      - TRANSCRIBE_DEVICE=cuda
    volumes:
      - ./models:/models
    ports:
      - "9000:9000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
```

### Standalone

1. Install Python 3.9-3.11 and ffmpeg
2. Download the repository files: `subgen.py`, `language_code.py`, `utils.py`, `requirements.txt`
3. Install dependencies: `pip install -r requirements.txt`
4. For GPU support, install CUDA Toolkit 12.3+
5. Run: `python3 subgen.py`

## Bazarr Configuration

1. In Bazarr, go to **Settings > Subtitles**
2. Under **Whisper Provider**, enter your Subgen endpoint: `http://<subgen-ip>:9000`
3. Save and test the connection

![Bazarr Whisper Configuration](https://wiki.bazarr.media/Additional-Configuration/images/whisper_config.png)

See [Bazarr Whisper Provider Documentation](https://wiki.bazarr.media/Additional-Configuration/Whisper-Provider/) for more details.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL` | `medium` | Model size: `tiny`, `base`, `small`, `medium`, `large-v1`, `large-v2`, `large-v3`, `large-v3-turbo`, `distil-large-v2`, `distil-large-v3`, `distil-large-v3.5` |
| `TRANSCRIBE_DEVICE` | `cpu` | Device for transcription: `cpu`, `cuda`, or `gpu` |
| `WHISPER_THREADS` | `4` | Number of CPU threads for computation |
| `COMPUTE_TYPE` | `auto` | CTranslate2 quantization type (see [docs](https://github.com/OpenNMT/CTranslate2/blob/master/docs/quantization.md)) |
| `WEBHOOK_PORT` | `9000` | Port for the ASR server |
| `MODEL_PATH` | `./models` | Directory to store downloaded models |
| `TEMP_FILE_PATH` | system temp | Directory for temporary files during upload (large files >1MB spill here) |
| `DEBUG` | `True` | Enable debug logging |
| `CLEAR_VRAM_ON_COMPLETE` | `True` | Free GPU memory when transcription queue is empty |
| `WORD_LEVEL_HIGHLIGHT` | `False` | Enable word-by-word timing in subtitles |
| `APPEND` | `False` | Add credit line to generated subtitles |
| `FORCE_DETECTED_LANGUAGE_TO` | `` | Override detected language (2-letter code, e.g., `en`, `fr`) |
| `DETECT_LANGUAGE_LENGTH` | `30` | Seconds of audio to analyze for language detection |
| `DETECT_LANGUAGE_OFFSET` | `0` | Start offset for language detection (skip intros) |
| `CUSTOM_REGROUP` | `cm_sl=84_sl=42++++++1` | Stable-TS segment regrouping pattern |
| `SUBGEN_KWARGS` | `{}` | Additional kwargs for Whisper model (advanced) |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/asr` | GET/POST | Transcribe audio to subtitles |
| `/detect-language` | GET/POST | Detect audio language |
| `/status` | GET | Get server version and status |
| `/` | GET | Root endpoint (redirects to /docs) |
| `/docs` | GET | Interactive API documentation |

## Testing with curl

You can test the API directly without Bazarr using curl.

**Check server status:**
```bash
curl http://localhost:9000/status
```

**Transcribe an audio file:**
```bash
curl -X POST "http://localhost:9000/asr?output=srt" \
  -F "audio_file=@/path/to/audio.mp3"
```

**Transcribe a video file (ffmpeg extracts audio):**
```bash
curl -X POST "http://localhost:9000/asr?language=en&output=srt" \
  -F "audio_file=@/path/to/video.mkv"
```

**Detect language:**
```bash
curl -X POST "http://localhost:9000/detect-language" \
  -F "audio_file=@/path/to/audio.mp3"
```

**All parameters:**
```bash
curl -X POST "http://localhost:9000/asr" \
  -F "audio_file=@/path/to/file.mp4" \
  -G -d "task=transcribe" \
  -d "language=en" \
  -d "output=srt" \
  -d "encode=true" \
  -d "word_timestamps=false"
```

You can also use the interactive Swagger UI at `http://localhost:9000/docs`.

## Docker Images

| Image | Description |
|-------|-------------|
| `mccloud/subgen:latest` | GPU/CUDA support (recommended) |
| `mccloud/subgen:cpu` | CPU-only (smaller image) |

## Supported Languages

Afrikaans, Arabic, Armenian, Azerbaijani, Belarusian, Bosnian, Bulgarian, Catalan, Chinese, Croatian, Czech, Danish, Dutch, English, Estonian, Finnish, French, Galician, German, Greek, Hebrew, Hindi, Hungarian, Icelandic, Indonesian, Italian, Japanese, Kannada, Kazakh, Korean, Latvian, Lithuanian, Macedonian, Malay, Marathi, Maori, Nepali, Norwegian, Persian, Polish, Portuguese, Romanian, Russian, Serbian, Slovak, Slovenian, Spanish, Swahili, Swedish, Tagalog, Tamil, Thai, Turkish, Ukrainian, Urdu, Vietnamese, and Welsh.

## Troubleshooting

- **High CPU usage when idle**: Try using the GPU image even on CPU-only systems
- **Out of memory errors**: Use a smaller model (`small` or `base`) or enable `CLEAR_VRAM_ON_COMPLETE`
- **Wrong language detected**: Use `FORCE_DETECTED_LANGUAGE_TO` to override

## Credits

- [OpenAI Whisper](https://github.com/openai/whisper) - Original ASR model
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - CTranslate2-based Whisper implementation
- [stable-ts](https://github.com/jianfch/stable-ts) - Subtitle timing refinement
- [Whisper ASR Webservice](https://github.com/ahmetoner/whisper-asr-webservice) - Bazarr webhook implementation reference

## License

See repository for license information.

## Support

- [GitHub Issues](https://github.com/McCloudS/subgen/issues)
- [PayPal Donation](https://www.paypal.com/donate/?hosted_button_id=SU4QQP6LH5PF6)
