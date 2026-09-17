"""Audio brand-extraction prompts (04 ``audio.brand_prompt``).

``m2`` is the pair in ``utils/prompts.py`` (imported, not copied).
``task2_italian`` is a verbatim copy of the improved prompt in
``task2-whisper-qwen.py``, which cannot be imported (hyphenated filename,
side effects at import).
"""

from __future__ import annotations

TASK2_TRANSCRIPT_SYSTEM_PROMPT = (
    "You are an expert in analyzing advertisements. "
    "You will be given a transcript from an advertisement video. "
    "Give me a thorough and complete list of ALL brand names mentioned in this "
    "transcript using your knowledge and the contextual information present."
)


def task2_transcript_user_prompt(transcript: str, language: str = "english") -> str:
    lang_hint = (
        f"The transcript below is in {language.capitalize()}. Whisper may have "
        "split or mis-spelled brand names phonetically — normalise them to their "
        "correct canonical spelling (e.g. 'very sure' → 'Verisure', "
        "'neo borosilina' → 'Neo Borocillina', 'monifarma' → 'Moneyfarm'). "
        "Ignore obvious transcription artefacts such as 'amara.org' (a known "
        "Whisper subtitle-watermark hallucination) and generic Italian nouns "
        "that are not brands (e.g. 'merluzzo' = cod)."
    )
    return f"""
You are an expert assistant tasked with extracting brand and product names from advertisement transcripts.

{lang_hint}

Your goal is to extract a newline-separated list of all proper brand or product names mentioned below.

Include: pharmaceutical brands, digital platforms, food/fashion brands, TV channels, any named brand entity.
Exclude: personal names, countries, cities, sports teams (unless branded), general nouns or slogans.
Do not include duplicates. Some brand names may have minor spelling errors.

Transcript:
\"\"\"{transcript.strip()}\"\"\"

List the brand names below, one per line:
"""


def brand_prompt_pair(variant: str, transcript: str, language: str | None) -> tuple[str, str]:
    """Return ``(system_prompt, user_prompt)`` for the configured variant."""
    if variant == "m2":
        from utils.prompts import QWEN_TRANSCRIPT_LOGO_PROMPT, QWEN_TRANSCRIPT_LOGO_SYSTEM_PROMPT
        return QWEN_TRANSCRIPT_LOGO_SYSTEM_PROMPT, QWEN_TRANSCRIPT_LOGO_PROMPT(transcript)
    if variant == "task2_italian":
        return TASK2_TRANSCRIPT_SYSTEM_PROMPT, task2_transcript_user_prompt(transcript, language or "italian")
    raise ValueError(f"unknown audio.brand_prompt variant {variant!r}")
