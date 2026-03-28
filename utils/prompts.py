


QWEN_TRANSCRIPT_LOGO_PROMPT = lambda transcript: f"""
You are an expert assistant tasked with extracting brand and product names from advertisement transcripts.

Your goal is to extract a **newline-separated list** of all proper brand or product names mentioned in the text below.

✅ Include:
- Pharmaceutical brands
- Digital platforms
- Food and fashion brands
- TV channels or media networks
- Any other **proper named brand entity**

🚫 Exclude:
- Personal names
- Countries or cities
- Sports teams or events (unless branded)
- General nouns or slogans

🧠 Notes:
- Do not include duplicates
- Some brand names may have minor spelling errors or be embedded in longer sentences

Transcript:
\"\"\"{transcript.strip()}\"\"\"

List the brand names below, one per line:
"""

# QWEN_TRANSCRIPT_LOGO_SYSTEM_PROMPT = "You are an expert in analyzing advertisements. You will be given an English transcript transcribed from an advertisement video in a non-English language. Your job is to carefully analyze this transcript and give me a thorough and complete list of ALL brand names mentioned in this transcript using both your knowledge, and contextual information present in the transcript."

QWEN_TRANSCRIPT_LOGO_SYSTEM_PROMPT = "You are an expert in analyzing advertisements. You will be given a transcript transcribed from an advertisement video in a non-English language. Your job is to carefully analyze this transcript and give me a thorough and complete list of ALL brand names mentioned in this transcript using both your knowledge, and contextual information present in the transcript."