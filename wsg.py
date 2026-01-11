wsg_version = '2026.01-bazarr'

"""
whisper_subtitle_generator - Bazarr-only Edition

This is a simplified version of Subgen that only provides Whisper ASR functionality for Bazarr. 
"""

import ast
import gc
import io
import logging
import os
import random
import shutil
import sys
import tempfile
import time
import faster_whisper
import ffmpeg
import torch
import stable_whisper

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, Union
from io import BytesIO

from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.responses import StreamingResponse
import numpy as np
from stable_whisper import Segment


from language_code import LanguageCode


def convert_to_bool(in_bool) -> bool:
    """
    Convert various input types to boolean.

    Args:
        in_bool: Value to convert (string, bool, int, etc.)

    Returns:
        True if the value represents a truthy string, False otherwise.
    """
    return str(in_bool).lower() in ('true', 'on', '1', 'y', 'yes')


@dataclass
class Config:
    """Application configuration loaded from environment variables."""

    whisper_model: str
    whisper_threads: int
    concurrent_transcriptions: int
    batch_size: int
    transcribe_device: str
    compute_type: str
    debug: bool
    webhook_port: int
    append: bool
    clear_vram_on_complete: bool
    word_level_highlight: bool
    model_location: str
    custom_regroup: str
    detect_language_length: int
    detect_language_offset: int
    force_detected_language_to: LanguageCode
    kwargs: dict[str, Any]
    temp_file_path: Optional[str]

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        device = os.getenv('TRANSCRIBE_DEVICE', 'cpu')
        if device == "gpu":
            device = "cuda"

        try:
            kwargs = ast.literal_eval(os.getenv('SUBGEN_KWARGS', '{}') or '{}')
        except ValueError:
            kwargs = {}
            logging.info("kwargs (SUBGEN_KWARGS) is an invalid dictionary, defaulting to empty '{}'")

        temp_path = os.getenv('TEMP_FILE_PATH', None)

        return cls(
            whisper_model=os.getenv('WHISPER_MODEL', 'small'),
            whisper_threads = int(os.getenv("WHISPER_THREADS", max(1, os.cpu_count() / 2))),
            concurrent_transcriptions = int(os.getenv('CONCURRENT_TRANSCRIPTIONS', 1)),
            batch_size = int(os.getenv('BATCH_SIZE', 1)),
            transcribe_device=device,
            compute_type=os.getenv('COMPUTE_TYPE', 'auto'),
            debug=convert_to_bool(os.getenv('DEBUG', False)),
            webhook_port=int(os.getenv('WEBHOOK_PORT', os.getenv('WEBHOOKPORT', 9000))),
            append=convert_to_bool(os.getenv('APPEND', False)),
            clear_vram_on_complete=convert_to_bool(os.getenv('CLEAR_VRAM_ON_COMPLETE', True)),
            word_level_highlight=convert_to_bool(os.getenv('WORD_LEVEL_HIGHLIGHT', False)),
            model_location=os.getenv('MODEL_PATH', './models'),
            custom_regroup=os.getenv('CUSTOM_REGROUP', 'cm_sl=84_sl=42++++++1'),
            detect_language_length=int(os.getenv('DETECT_LANGUAGE_LENGTH', 600)),
            detect_language_offset=int(os.getenv('DETECT_LANGUAGE_OFFSET', 120)),
            force_detected_language_to=LanguageCode.from_string(os.getenv('FORCE_DETECTED_LANGUAGE_TO', '')),
            kwargs=kwargs,
            temp_file_path=temp_path,
        )


# Load configuration
config = Config.from_env()

# Configure temp file directory for large uploads
if config.temp_file_path:
    os.makedirs(config.temp_file_path, exist_ok=True)
    tempfile.tempdir = config.temp_file_path

# FastAPI app
app = FastAPI()
model = None

in_docker = os.path.exists('/.dockerenv')
docker_status = "Docker" if in_docker else "Standalone"


class MultiplePatternsFilter(logging.Filter):
    """Filter to hide common logging we don't want to see."""
    def filter(self, record):
        patterns = [
            "Compression ratio threshold is not met",
            "Processing segment at",
            "Log probability threshold is",
            "Reset prompt",
            "Attempting to release",
            "released on ",
            "Attempting to acquire",
            "acquired on",
            "header parsing failed",
            "timescale not set",
            "misdetection possible",
        ]
        return not any(pattern in record.getMessage() for pattern in patterns)


# Configure logging
level = logging.DEBUG if config.debug else logging.INFO
logging.basicConfig(stream=sys.stderr, level=level, format="%(asctime)s %(levelname)s: %(message)s")

logger = logging.getLogger()
logger.setLevel(level)

for handler in logger.handlers:
    handler.addFilter(MultiplePatternsFilter())

logging.getLogger("multipart").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)

last_print_time = None


def progress(seek, total):
    """Force flush to print progress correctly in Docker."""
    sys.stdout.flush()
    sys.stderr.flush()
    if docker_status == 'Docker':
        global last_print_time
        current_time = time.time()
        if last_print_time is None or (current_time - last_print_time) >= 5:
            last_print_time = current_time
            logging.info("")


TIME_OFFSET = 5


def log_with_context(
    level: int,
    message: str,
    context: Optional[str] = None,
    context_prefix: str = "for",
) -> None:
    """
    Log a message with optional context information.

    Simplifies the common pattern of conditionally including file paths
    or other context in log messages.

    Args:
        level: Logging level (e.g., logging.INFO, logging.ERROR)
        message: Base message to log
        context: Optional context to append (e.g., file path)
        context_prefix: Prefix before context (default: "for")
    """
    if context:
        full_message = f"{message} {context_prefix} '{context}'"
    else:
        full_message = message
    logging.log(level, full_message)


def append_line(result) -> None:
    """Append credit line to transcription result."""
    if not config.append:
        return

    last_segment = result.segments[-1]
    date_time_str = datetime.now().strftime("%d %b %Y - %H:%M:%S")
    appended_text = f"Transcribed by whisperAI with faster-whisper ({config.whisper_model}) on {date_time_str}"

    new_segment = Segment(
        start=last_segment.start + TIME_OFFSET,
        end=last_segment.end + TIME_OFFSET,
        text=appended_text,
        words=[],
        id=last_segment.id + 1,
    )
    result.segments.append(new_segment)


def start_model() -> None:
    """Load the Whisper model into memory."""
    global model
    if model is not None:
        return

    logging.debug("Loading Whisper model...")
    model = stable_whisper.load_faster_whisper(
        config.whisper_model,
        download_root=config.model_location,
        device=config.transcribe_device,
        cpu_threads=config.whisper_threads,
        num_workers=config.concurrent_transcriptions,
        compute_type=config.compute_type,
    )


def delete_model() -> None:
    """Unload the Whisper model from memory if configured."""
    global model
    if not config.clear_vram_on_complete:
        return

    if model is not None:
        logging.debug("Clearing model from memory...")
        try:
            # Unload the underlying faster-whisper model
            model.model.unload_model()
        except Exception as e:
            logging.debug(f"Error unloading model: {e}")

        del model
        model = None

    # Clear CUDA cache if using GPU
    if config.transcribe_device.lower() == 'cuda' and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logging.debug("CUDA cache cleared.")

    # Aggressive garbage collection - run multiple generations
    gc.collect(0)
    gc.collect(1)
    gc.collect(2)

    # On Linux, return freed memory to OS using malloc_trim
    if sys.platform == 'linux':
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
            logging.debug("Memory returned to OS via malloc_trim.")
        except Exception as e:
            logging.debug(f"malloc_trim not available: {e}")


def save_upload_to_temp(upload_file: UploadFile, temp_dir: Optional[str] = None) -> str:
    """
    Save uploaded file to a temp file without loading entirely into memory.

    Uses shutil.copyfileobj to stream the file to disk.

    Args:
        upload_file: FastAPI UploadFile object
        temp_dir: Directory for temp file (default: system temp or TEMP_FILE_PATH)

    Returns:
        Path to the saved temp file (caller must delete)
    """
    suffix = os.path.splitext(upload_file.filename or '')[1] or '.tmp'
    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
        dir=temp_dir or config.temp_file_path,
    )
    try:
        upload_file.file.seek(0)
        shutil.copyfileobj(upload_file.file, temp_file)
        temp_file.close()
        logging.debug(f"Saved upload to temp file: {temp_file.name}")
        return temp_file.name
    except Exception as e:
        temp_file.close()
        os.unlink(temp_file.name)
        raise e


def handle_multiple_audio_tracks(file_path: str, language: LanguageCode | None = None) -> BytesIO | None:
    """
    Handles the possibility of a media file having multiple audio tracks.
    
    If the media file has multiple audio tracks, it will extract the audio track of the selected language. Otherwise, it will extract the first audio track.
    
    Parameters:
    file_path (str): The path to the media file.
    language (LanguageCode | None): The language of the audio track to search for. If None, it will extract the first audio track.
    
    Returns:
    io.BytesIO  | None: The audio or None if no audio track was extracted.
    """
    audio_bytes = None
    audio_tracks = get_audio_tracks(file_path)

    if len(audio_tracks) > 1:
        logging.debug(f"Handling multiple audio tracks from {file_path} and planning to extract audio track of language {language}")
        logging.debug(
            "Audio tracks:\n"
            + "\n".join([f"  - {track['index']}: {track['codec']} {track['language']} {('default' if track['default'] else '')}" for track in audio_tracks])
        )

        if language is not None:
            audio_track = get_audio_track_by_language(audio_tracks, language)
        if audio_track is None:
            audio_track = audio_tracks[0]
        
        audio_bytes = extract_audio_track_to_memory(file_path, audio_track["index"])
        if audio_bytes is None:
            logging.error(f"Failed to extract audio track {audio_track['index']} from {file_path}")
            return None
    return audio_bytes


def get_audio_tracks(video_file):
    """
    Extracts information about the audio tracks in a file.

    Returns:
        List of dictionaries with information about each audio track.
        Each dictionary has the following keys:
            index (int): The stream index of the audio track.
            codec (str): The name of the audio codec.
            channels (int): The number of audio channels.
            language (LanguageCode): The language of the audio track.
            title (str): The title of the audio track.
            default (bool): Whether the audio track is the default for the file.
            forced (bool): Whether the audio track is forced.
            original (bool): Whether the audio track is the original.
            commentary (bool): Whether the audio track is a commentary.

    Raises:
        ffmpeg.Error: If FFmpeg fails to probe the file.
    """
    try:
        # Probe the file to get audio stream metadata
        probe = ffmpeg.probe(video_file, select_streams='a')
        audio_streams = probe.get('streams', [])
        
        # Extract information for each audio track
        audio_tracks = []
        for stream in audio_streams:
            audio_track = {
                "index": int(stream.get("index", None)),
                "codec": stream.get("codec_name", "Unknown"),
                "channels": int(stream.get("channels", None)),
                "language": LanguageCode.from_iso_639_2(stream.get("tags", {}).get("language", "Unknown")),
                "title": stream.get("tags", {}).get("title", "None"),
                "default": stream.get("disposition", {}).get("default", 0) == 1,
                "forced": stream.get("disposition", {}).get("forced", 0) == 1,
                "original": stream.get("disposition", {}).get("original", 0) == 1,
                "commentary": "commentary" in stream.get("tags", {}).get("title", "").lower()
            }
            audio_tracks.append(audio_track)    
        return audio_tracks

    except ffmpeg.Error as e:
        logging.error(f"FFmpeg error: {e.stderr}")
        return []
    except Exception as e:
        logging.error(f"An error occurred while reading audio track information: {str(e)}")
        return []


def extract_audio_to_temp(
    input_path: str,
    temp_dir: Optional[str] = None,
    start_time: Optional[int] = None,
    duration: Optional[int] = None,
) -> str:
    """
    Extract audio from video/audio file to a temp WAV file.

    Uses ffmpeg to stream-extract audio without loading video into memory.

    Args:
        input_path: Path to input video/audio file
        temp_dir: Directory for temp file (default: system temp or TEMP_FILE_PATH)
        start_time: Optional start time in seconds
        duration: Optional duration in seconds

    Returns:
        Path to the extracted audio WAV file (caller must delete)
    """
    temp_audio = tempfile.NamedTemporaryFile(
        delete=False,
        suffix='.wav',
        dir=temp_dir or config.temp_file_path,
    )
    temp_audio.close()

    try:
        input_kwargs = {}
        if start_time is not None:
            input_kwargs['ss'] = start_time
        if duration is not None:
            input_kwargs['t'] = duration

        stream = ffmpeg.input(input_path, **input_kwargs)
        stream = stream.output(
            temp_audio.name,
            format='wav',
            acodec='pcm_s16le',
            ac=1,
            ar=16000,
            vn=None,  # Skip video streams entirely
            threads=1,  # Limit thread usage to reduce memory
            **{'max_alloc': '104857600'}  # 100MB max buffer allocation
        )
        stream.run(overwrite_output=True, capture_stderr=True, quiet=True)

        logging.debug(f"Extracted audio to temp file: {temp_audio.name}")
        return temp_audio.name

    except ffmpeg.Error as e:
        os.unlink(temp_audio.name)
        logging.error(f"FFmpeg error extracting audio: {e.stderr.decode() if e.stderr else str(e)}")
        raise
    except Exception as e:
        os.unlink(temp_audio.name)
        logging.error(f"Error extracting audio: {str(e)}")
        raise


def extract_audio_segment_to_memory(input_file, start_time, duration):
    """
    Extract a segment of audio from input_file, starting at start_time for duration seconds.
    
    :param input_file: UploadFile object or path to the input audio file
    :param start_time: Start time in seconds (e.g., 60 for 1 minute)
    :param duration: Duration in seconds (e.g., 30 for 30 seconds)
    :return: BytesIO object containing the audio segment
    """
    try:
        if hasattr(input_file, 'file') and hasattr(input_file.file, 'read'):  # Handling UploadFile
            input_file.file.seek(0)  # Ensure the file pointer is at the beginning
            input_stream = 'pipe:0'
            input_kwargs = {'input': input_file.file.read()}
        elif isinstance(input_file, str):  # Handling local file path
            input_stream = input_file
            input_kwargs = {}
        else:
            raise ValueError("Invalid input: input_file must be a file path or an UploadFile object.")

        logging.info(f"Extracting audio from: {input_stream}, start_time: {start_time}, duration: {duration}")

        # Run FFmpeg to extract the desired segment
        out, _ = (
            ffmpeg
            .input(input_stream, ss=start_time, t=duration)  # Set start time and duration
            .output('pipe:1', format='wav', acodec='pcm_s16le', ar=16000)  # Output to pipe as WAV
            .run(capture_stdout=True, capture_stderr=True, **input_kwargs)
        )

        # Check if the output is empty or null
        if not out:
            raise ValueError("FFmpeg output is empty, possibly due to invalid input.")
        
        return io.BytesIO(out)  # Convert output to BytesIO for in-memory processing

    except ffmpeg.Error as e:
        logging.error(f"FFmpeg error: {e.stderr.decode()}")
        return None
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return None

    except ffmpeg.Error as e:
        logging.error(f"FFmpeg error: {e.stderr.decode()}")
        return None
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return None


async def get_audio_chunk(audio_file, offset=config.detect_language_offset, length=config.detect_language_length, sample_rate=16000, audio_format=np.int16):
    """
    Extract a chunk of audio from a file, starting at the given offset and of the given length.
    
    :param audio_file: The audio file (UploadFile or file-like object).
    :param offset: The offset in seconds to start the extraction.
    :param length: The length in seconds for the chunk to be extracted.
    :param sample_rate: The sample rate of the audio (default 16000).
    :param audio_format: The audio format to interpret (default int16, 2 bytes per sample).
    
    :return: A numpy array containing the extracted audio chunk.
    """
    
    # Number of bytes per sample (for int16, 2 bytes per sample)
    bytes_per_sample = np.dtype(audio_format).itemsize
    
    # Calculate the start byte based on offset and sample rate
    start_byte = offset * sample_rate * bytes_per_sample
    
    # Calculate the length in bytes based on the length in seconds
    length_in_bytes = length * sample_rate * bytes_per_sample
    
    # Seek to the start position (this assumes the audio_file is a file-like object)
    await audio_file.seek(start_byte)
    
    # Read the required chunk of audio (length_in_bytes)
    chunk = await audio_file.read(length_in_bytes)
    
    # Convert the chunk into a numpy array (normalized to float32)
    audio_data = np.frombuffer(chunk, dtype=audio_format).flatten().astype(np.float32) / 32768.0
    
    return audio_data


@app.get("/asr")
@app.get("/detect-language")
def handle_get_request(request: Request) -> dict[str, str]:
    return {"error": "Use POST request. See https://github.com/McCloudS/subgen for configuration."}


@app.get("/")
def webui() -> dict[str, str]:
    return {"message": "Subgen Bazarr Edition - Use /asr or /detect-language endpoints"}


@app.get("/status")
def status() -> dict[str, str]:
    return {
        "version": f"whisper_subtitle_generator {wsg_version}, stable-ts {stable_whisper.__version__}, faster-whisper {faster_whisper.__version__} ({docker_status})"
    }


@app.post("//asr")
@app.post("/asr")
async def asr(
    task: Union[str, None] = Query(default="transcribe", enum=["transcribe", "translate"]),
    language: Union[str, None] = Query(default=None),
    video_file: Union[str, None] = Query(default=None),
    initial_prompt: Union[str, None] = Query(default=None),
    audio_file: UploadFile = File(...),
    encode: bool = Query(default=True, description="Encode audio first through ffmpeg"),
    output: Union[str, None] = Query(default="srt", enum=["txt", "vtt", "srt", "tsv", "json"]),
    word_timestamps: bool = Query(default=False, description="Word-level timestamps"),
):
    """
    Transcribe or translate audio file to subtitles.

    This endpoint is used by Bazarr to generate subtitles from audio.
    Files are streamed to disk to avoid loading large videos into memory.
    """
    result = None
    temp_upload_path = None
    temp_audio_path = None

    try:
        log_with_context(
            logging.INFO,
            f"{task.capitalize()} from Bazarr/ASR webhook",
            video_file,
            context_prefix="of file",
        )

        if config.force_detected_language_to:
            language = config.force_detected_language_to.to_iso_639_1()
            logging.info(f"Forcing detected language to {config.force_detected_language_to}")

        start_time = time.time()

        # Save upload to temp file (streams to disk, doesn't load into RAM)
        #temp_upload_path = save_upload_to_temp(audio_file)
        #logging.debug(f"Upload saved to: {temp_upload_path}")
        #await audio_file.close()

        start_model()

        args = {'progress_callback': progress}
        args['batch_size'] = config.batch_size
        logging.info(f"Processing with batch_size={config.batch_size}")

        args['vad_filter'] = True
        args['temperature'] = 0
        args['beam_size'] = 5
        args['no_speech_threshold'] = 0.8
        args['condition_on_previous_text'] = False
        args['suppress_blank'] = True
        args['vad_parameters'] = dict(min_silence_duration_ms=500)

        file_content = audio_file.file.read()

        # Extract audio to temp WAV file (ffmpeg streams from disk)
        #temp_audio_path = extract_audio_to_temp(temp_upload_path)
        #args['audio'] = temp_audio_path

        if encode:
            args['audio'] = file_content
        else:
            args['audio'] = np.frombuffer(file_content, np.int16).flatten().astype(np.float32) / 32768.0
            args['input_sr'] = 16000

        if config.custom_regroup:
            args['regroup'] = config.custom_regroup

        args.update(config.kwargs)

        result = model.transcribe(task=task, language=language, **args)
        append_line(result)

        elapsed_time = time.time() - start_time
        minutes, seconds = divmod(int(elapsed_time), 60)
        log_with_context(
            logging.INFO,
            f"{task.capitalize()} complete in {minutes}m {seconds}s",
            video_file,
            context_prefix="for",
        )

        # Convert result to string immediately to release transcription data
        if result:
            srt_content = result.to_srt_vtt(filepath=None, word_level=config.word_level_highlight)
            # Explicitly delete result to free memory before cleanup
            del result
            result = None
            response = StreamingResponse(
                iter(srt_content),
                media_type="text/plain",
                headers={'Source': f'{task.capitalize()}d using stable-ts from Subgen!'},
            )
        else:
            response = None

    except Exception as e:
        log_with_context(
            logging.ERROR,
            f"Error processing Bazarr file: {e}",
            video_file,
            context_prefix="for",
        )
        response = None

    finally:
        await audio_file.close()
        # Clean up temp files
        if temp_upload_path and os.path.exists(temp_upload_path):
            os.unlink(temp_upload_path)
            logging.debug(f"Cleaned up temp upload: {temp_upload_path}")
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)
            logging.debug(f"Cleaned up temp audio: {temp_audio_path}")
        # Clean up any lingering result object
        if result is not None:
            del result
        delete_model()

    return response


@app.post("//detect-language")
@app.post("/detect-language")
async def detect_language(
    audio_file: UploadFile = File(...),
    encode: bool = Query(default=True, description="Encode audio first through ffmpeg"),
    video_file: Union[str, None] = Query(default=None),
    detect_lang_length: int = Query(default=None, description="Seconds to analyze"),
    detect_lang_offset: int = Query(default=None, description="Start offset in seconds"),
):
    """
    Detect the language of an audio file.

    This endpoint is used by Bazarr to determine what language subtitle to generate.
    Files are streamed to disk to avoid loading large videos into memory.
    """
    # Use config defaults if not provided
    lang_length = detect_lang_length if detect_lang_length is not None else config.detect_language_length
    lang_offset = detect_lang_offset if detect_lang_offset is not None else config.detect_language_offset

    if config.force_detected_language_to:
        logging.debug(f"Skipping detection, forced to {config.force_detected_language_to.to_name()}")
        return {
            "detected_language": config.force_detected_language_to.to_name(),
            "language_code": config.force_detected_language_to.to_iso_639_1(),
        }

    detected_language = LanguageCode.NONE
    language_code = 'und'
    temp_upload_path = None
    temp_audio_path = None

    try:
        log_with_context(
            logging.INFO,
            "Detecting language from Bazarr",
            video_file,
            context_prefix="for",
        )

        # Save upload to temp file (streams to disk, doesn't load into RAM)
        temp_upload_path = save_upload_to_temp(audio_file)

        start_model()

        audio_file.file.seek(0)

        args = {'progress_callback': progress}
        args['batch_size'] = config.batch_size
        logging.info(f"Processing with batch_size={config.batch_size}")

        # Extract only the needed audio segment to temp file
        #temp_audio_path = extract_audio_to_temp(
        #    temp_upload_path,
        #    start_time=lang_offset,
        #    duration=lang_length,
        #)
        #args['audio'] = temp_audio_path

        if encode:
            args['audio'] = extract_audio_segment_to_memory(audio_file, lang_offset, lang_length).read()
            args['input_sr'] = 16000
        else:
            args['audio'] = await get_audio_chunk(audio_file, lang_offset, lang_length)
            args['input_sr'] = 16000

        args.update(config.kwargs)
        # Get language from transcription result without storing full result
        transcribe_result = model.transcribe(**args)
        detected_language = LanguageCode.from_name(transcribe_result.language)
        language_code = detected_language.to_iso_639_1()
        # Explicitly delete transcription result
        del transcribe_result
        logging.debug(f"Detected: {detected_language.to_name()} ({language_code})")

    except Exception as e:
        log_with_context(
            logging.ERROR,
            f"Error detecting language: {e}",
            video_file,
            context_prefix="for",
        )

    finally:
        await audio_file.close()
        # Clean up temp files
        if temp_upload_path and os.path.exists(temp_upload_path):
            os.unlink(temp_upload_path)
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)
        delete_model()

    return {"detected_language": detected_language.to_name(), "language_code": language_code}


if __name__ == "__main__":
    import uvicorn

    logging.info(f"whisper_subtitle_generator v{wsg_version} (Bazarr Edition)")
    logging.info(
        f"Threads: {config.whisper_threads}, "
        f"Device: {config.transcribe_device}, "
        f"Model: {config.whisper_model}"
    )
    if config.temp_file_path:
        logging.info(f"Temp file path: {config.temp_file_path}")
    else:
        logging.info(f"Temp file path: {tempfile.gettempdir()} (system default)")
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    uvicorn.run("__main__:app", host="0.0.0.0", port=config.webhook_port, use_colors=True)
