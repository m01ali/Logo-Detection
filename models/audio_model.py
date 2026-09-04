import os
import json
import argparse
from typing import List, Dict
import torch
try:
    # moviepy 2.x exposes these at the top level
    from moviepy import AudioFileClip, VideoFileClip
except ImportError:
    # moviepy 1.x keeps them under moviepy.editor
    from moviepy.editor import AudioFileClip, VideoFileClip
try:
    import torchaudio
except Exception:
    torchaudio = None
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import pandas as pd
import librosa
from pathlib import Path

# class WhisperModel:
#     def __init__(self, model_id='openai/whisper-large-v3'):
#         self.model_id = model_id
#         self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
#         torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

#         self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
#             model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
#         )
#         self.model.to(self.device)

#         self.processor = AutoProcessor.from_pretrained(model_id)

#         self.pipe = pipeline(
#             "automatic-speech-recognition",
#             model=self.model,
#             tokenizer=self.processor.tokenizer,
#             feature_extractor=self.processor.feature_extractor,
#             torch_dtype=torch_dtype,
#             device=self.device,
#             # chunk_length_s=15,
#         )

#     def run_example(self, audio_path:str, task:str = 'translate', **kwargs):
#         result = self.pipe(
#             audio_path,
#             return_timestamps=True,
#             chunk_length_s=30,
#             stride_length_s=5,
#             batch_size=8,
#             generate_kwargs = {
#                 'language': 'english',
#                 'task': task,
#                 #'temperature': (0.0, 0.2, 0.4, 0.6, 0.8, 0.1),
#                 #'compression_ratio_threshold': 1.35,
#                 #'logprob_threshold': -1.0,
#                 # 'no_speech_threshold': 0.6,
#                 # 'condition_on_prev_tokens': False,
#             }
#         )
#         chunks_df = pd.DataFrame(result['chunks'])
#         transcript = result['text']
#         return transcript, chunks_df
    
# class WhisperModel:
#     """
#     Wrapper around HuggingFace Whisper v3 large for long-form
#     transcription and translation (Italian→English by default).
#     """

#     def __init__(
#         self,
#         model_id: str = "openai/whisper-large-v3",
#         chunk_length_s: int = 30,
#         stride_length_s: int = None,
#         device: str = None,
#         dtype: torch.dtype = None,
#     ):
#         """
#         Initialize the WhisperModel.

#         Args:
#             model_id (str): HuggingFace model identifier.
#             chunk_length_s (int): Segment length (seconds) for chunked processing.
#             stride_length_s (int, optional): Overlap (seconds) between chunks.
#                 If None, defaults to chunk_length_s // 6.
#             device (str, optional): "cuda" or "cpu". If None, auto-detects.
#             dtype (torch.dtype, optional): torch.float16 or torch.float32.
#                 If None, uses float16 on GPU or float32 on CPU.
#         """
#         # 1️⃣ Determine compute device
#         self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
#         # 2️⃣ Determine tensor dtype
#         self.dtype = dtype or (torch.float16 if "cuda" in self.device else torch.float32)

#         # 3️⃣ Chunking parameters
#         self.chunk_length_s = chunk_length_s
#         # default stride = chunk_length_s / 6 for smooth overlap
#         self.stride_length_s = stride_length_s if stride_length_s is not None else chunk_length_s // 6

#         # 4️⃣ Load the model with low-memory options
#         self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
#             model_id,
#             torch_dtype=self.dtype,
#             low_cpu_mem_usage=True,
#             use_safetensors=True
#         ).to(self.device)

#         # 5️⃣ Load processor (feature extractor + tokenizer)
#         self.processor = AutoProcessor.from_pretrained(model_id)

#         # 6️⃣ Build the pipeline: chunked translation into English
#         self.asr_pipeline = pipeline(
#             task="automatic-speech-recognition",
#             model=self.model,
#             tokenizer=self.processor.tokenizer,
#             feature_extractor=self.processor.feature_extractor,
#             chunk_length_s=self.chunk_length_s,
#             stride_length_s=(self.stride_length_s, self.stride_length_s),
#             batch_size=4,
#             device=self.device,
#             torch_dtype=self.dtype,
#             return_timestamps=True,
#             generate_kwargs = {
#                 'language': 'english',
#                 'task': "translate",
#                 'temperature': (0.0, 0.2, 0.4, 0.6, 0.8, 0.1),
#                 'compression_ratio_threshold': 1.35,
#                 'logprob_threshold': -1.0,
#                 # 'no_speech_threshold': 0.6,
#                 # 'condition_on_prev_tokens': False,
#             }        
#         )

#     def run_example(self, audio_path: str) -> str:
#         """
#         Transcribe and translate a long-form audio file.

#         Args:
#             audio_path (str): Path to the audio file (e.g., .mp3, .wav).

#         Returns:
#             str: The full English transcript.
#         """
#         # Run the pipeline; it will chunk, translate, then stitch results
#         result = self.asr_pipeline(audio_path)
#         chunks_df = pd.DataFrame(result['chunks'])
#         transcript = result['text']
#         return transcript, chunks_df

import torch
import pandas as pd
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


class WhisperModel:
    """
    Wrapper around HuggingFace Whisper v3 large for long-form
    transcription (original language) and translation (to English).

    Minimal-change version with per-call control:
      - mode: "transcribe" (same language) or "translate" (to English)
      - language: None/"italian"/"english"/...
      - return_language: optionally return detected language (when supported)

    Backward compatible:
      - run_example(..., return_language=False) -> (transcript, chunks_df)
      - run_example(..., return_language=True)  -> (transcript, chunks_df, lang)
    """

    def __init__(
        self,
        model_id: str = "openai/whisper-large-v3",
        chunk_length_s: int = 30,
        stride_length_s: int | None = None,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        batch_size: int = 4,
    ):
        # 1) Determine compute device / dtype
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.dtype = dtype or (torch.float16 if self.device != "cpu" else torch.float32)

        # 2) Chunking parameters
        self.chunk_length_s = int(chunk_length_s)
        self.stride_length_s = (
            int(stride_length_s) if stride_length_s is not None else max(1, self.chunk_length_s // 6)
        )
        self.batch_size = int(batch_size)

        # 3) Load model & processor
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(model_id)

        # 4) Build a single ASR pipeline; task/language are set per call
        self.asr_pipeline = pipeline(
            task="automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            chunk_length_s=self.chunk_length_s,
            stride_length_s=(self.stride_length_s, self.stride_length_s),
            batch_size=self.batch_size,
            device=self.device,
            torch_dtype=self.dtype,
            return_timestamps=True,
        )

    def run_example(
        self,
        audio_path: str,
        mode: str = "translate",            # "translate" or "transcribe"
        language: str | None = "english",   # e.g., "italian" for transcribe; "english" for translate; None = auto
        return_language: bool = False,
    ):
        """
        Transcribe or translate a long-form audio file with chunking.

        Args:
            audio_path: path to audio (e.g., .wav)
            mode: "translate" (to English) or "transcribe" (same language)
            language: target/source language hint. For Italian originals, use language="italian" when transcribing.
            return_language: if True, attempts to return detected language code (when supported by pipeline)

        Returns:
            If return_language=False:
                (transcript: str, chunks_df: pd.DataFrame)
            If return_language=True:
                (transcript: str, chunks_df: pd.DataFrame, lang: str | None)
        """
        mode = (mode or "translate").lower()
        if mode not in ("translate", "transcribe"):
            raise ValueError(f"WhisperModel.run_example: invalid mode='{mode}', expected 'translate' or 'transcribe'.")

        # Per-call control over Whisper behavior
        gen_kwargs = {
            "task": mode,  # "translate" or "transcribe",
            'temperature': (0.0, 0.2, 0.4, 0.6, 0.8, 0.1),
            'compression_ratio_threshold': 1.35,
            'logprob_threshold': -1.0,
        }
        if language:
            gen_kwargs["language"] = language.lower()

        # Some transformers versions accept return_language; be defensive
        try:
            result = self.asr_pipeline(
                audio_path,
                generate_kwargs=gen_kwargs,
                return_language=return_language,
            )
        except TypeError:
            # Fallback if return_language is not supported
            result = self.asr_pipeline(
                audio_path,
                generate_kwargs=gen_kwargs,
            )

        # Extract outputs safely
        transcript = result.get("text", "") if isinstance(result, dict) else ""
        chunks = result.get("chunks", []) if isinstance(result, dict) else []
        chunks_df = pd.DataFrame(chunks)

        lang = None
        if return_language:
            # Different transformers versions may structure language differently
            maybe_lang = result.get("language") if isinstance(result, dict) else None
            if isinstance(maybe_lang, str):
                lang = maybe_lang.lower()
            elif isinstance(maybe_lang, dict):
                # e.g., {"detected_language":"it", "score": ...} or {"language":"it"}
                lang = (
                    maybe_lang.get("language")
                    or maybe_lang.get("detected_language")
                    or maybe_lang.get("lang")
                )
                if isinstance(lang, str):
                    lang = lang.lower()

        if return_language:
            return transcript, chunks_df, lang
        return transcript, chunks_df

    
def convert_video_to_audio(video_path:str):
    with VideoFileClip(video_path) as video_clip:
        audio_clip = video_clip.audio
        audio_clip.write_audiofile('logo_audio.wav')


if __name__ == "__main__":
    video_path = Path('../videos/1.mp4')
    print('here')
    # convert_video_to_audio(video_path)

    audio_path = '../logo_audio.wav'

    model = WhisperModel()

    transcript, chunk_df, lang = model.run_example(audio_path, 
                                             mode='transcribe',
                                             language='italian',
                                             return_language=True)
    print(transcript, lang)