"""
Topic: Audio transcription pipelines (Whisper and equivalents)

Speech-to-text models (whisper-1, gpt-4o-transcribe, Deepgram, AssemblyAI,
...) turn audio into text. LiteLLM gives them one call:

    litellm.transcription(model=AUDIO_MODEL, file=open("clip.mp3", "rb"))

A transcript alone is rarely the product. A real pipeline looks like:

  audio -> [chunk if long] -> transcribe (+ vocabulary hint)
        -> timestamps -> LLM post-processing -> structured output

The sample "meeting" is generated with text-to-speech, so the exact script
is known and the transcript can be scored against it. It's cached in
samples/, so TTS runs only once.

Run:  python 03_multimodal/10_audio_transcription.py
      python 03_multimodal/10_audio_transcription.py path/to/audio.mp3   (your own file)
"""

import re
import sys
from pathlib import Path

import litellm
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import AUDIO_MODEL, TTS_MODEL, banner, chat, show

SAMPLES = Path(__file__).parent / "samples"

# A long recording is split into chunks before transcribing. Here the two
# chunks are generated separately, standing in for a split recording.
MEETING_CHUNKS = [
    "Okay, quick sync on the Qlaro launch. Anjali, the PgBouncer rollout for "
    "the billing database is done, right? Great. Then the main blocker is the "
    "Zephyrine dashboard. Rohit, can you fix the latency alerts by Thursday?",
    "Sure, I'll take the alerts. Also, Anjali will write the Qlaro release "
    "notes by Friday. We decided to delay the mobile beta by one week, so "
    "the new date is October ninth. Let's meet again on Monday.",
]
# Names and jargon that a general speech model has never seen.
KEY_TERMS = ["Qlaro", "PgBouncer", "Zephyrine", "Anjali", "Rohit"]
# Whisper's `prompt` is a text hint that biases spelling and style. It is NOT
# an instruction, so write it like a sample of the expected transcript.
GLOSSARY = "Meeting notes. Attendees: Anjali, Rohit. Topics: Qlaro launch, PgBouncer, Zephyrine dashboard."


class ActionItem(BaseModel):
    owner: str
    task: str
    due: str | None


class MeetingNotes(BaseModel):
    summary: str
    decisions: list[str]
    action_items: list[ActionItem]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def credentials(model: str) -> dict:
    """Key and base URL LiteLLM resolves for this model (e.g. LITELLM_PROXY_API_KEY).

    litellm.speech() only looks for OPENAI_API_KEY, even for litellm_proxy/
    models, so pass them explicitly. For direct providers both are None and
    LiteLLM falls back to its usual env vars.
    """
    _, _, api_key, api_base = litellm.get_llm_provider(model)
    return {"api_key": api_key, "api_base": api_base}


def make_sample_audio() -> list[Path]:
    """Generate the meeting chunks with TTS, once."""
    SAMPLES.mkdir(exist_ok=True)
    paths = []
    for i, text in enumerate(MEETING_CHUNKS, start=1):
        path = SAMPLES / f"meeting_part{i}.mp3"
        if not path.exists():
            print(f"Generating {path.name} with {TTS_MODEL}...")
            audio = litellm.speech(model=TTS_MODEL, input=text, voice="alloy", **credentials(TTS_MODEL))
            path.write_bytes(audio.content)
        paths.append(path)
    return paths


def transcribe(path: Path, **kwargs):
    with open(path, "rb") as f:
        return litellm.transcription(model=AUDIO_MODEL, file=f, **credentials(AUDIO_MODEL), **kwargs)


def term_score(text: str) -> str:
    found = [t for t in KEY_TERMS if re.search(rf"\b{t}\b", text, re.IGNORECASE)]
    missed = [t for t in KEY_TERMS if t not in found]
    return f"{len(found)}/{len(KEY_TERMS)} key terms correct" + (f", missed: {missed}" if missed else "")


def field(segment, name: str):
    """Segments come back as dicts or objects depending on the provider."""
    return segment.get(name) if isinstance(segment, dict) else getattr(segment, name)


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------
def demo_plain_vs_prompted(clip: Path) -> None:
    banner("1. Plain transcription vs a vocabulary hint")
    plain = transcribe(clip).text
    show("Plain", plain)
    print(term_score(plain))

    hinted = transcribe(clip, prompt=GLOSSARY).text
    show("With vocabulary prompt", hinted)
    print(term_score(hinted))
    # Product names and people's names are what users notice first. A glossary
    # from your domain (CRM names, product catalogue) is a cheap, big win.


def demo_timestamps(clip: Path) -> None:
    banner("2. Timestamps with verbose_json")
    try:
        result = transcribe(clip, response_format="verbose_json", timestamp_granularities=["segment"])
    except Exception as e:
        # Only some models support this (whisper-1 does; gpt-4o-transcribe doesn't).
        print(f"{AUDIO_MODEL} doesn't support verbose_json: {type(e).__name__}: {str(e)[:150]}")
        return
    segments = result.get("segments") or []
    if not segments:
        print("No segments returned; this model only gives plain text.")
        return
    for seg in segments:
        print(f"  [{field(seg, 'start'):6.2f}s - {field(seg, 'end'):6.2f}s] {field(seg, 'text').strip()}")
    # Timestamps let you link notes back to the recording ("jump to 0:42"),
    # generate subtitles, and line up speakers (diarization is a separate step).


def demo_chunked(clips: list[Path]) -> str:
    banner("3. Long audio: transcribe chunk by chunk, carrying context")
    # APIs cap uploads (about 25 MB for OpenAI). Real pipelines split long audio
    # with ffmpeg/pydub, ideally at silences so no word is cut in half.
    # Passing the end of the previous chunk as the prompt keeps names and
    # style consistent across chunk boundaries. It also carries mistakes
    # forward, so keep the glossary in front of it.
    parts: list[str] = []
    for clip in clips:
        context = " ".join(parts)[-200:]  # tail of the transcript so far
        parts.append(transcribe(clip, prompt=f"{GLOSSARY} {context}").text.strip())
        print(f"  {clip.name}: {len(parts[-1])} chars")
    transcript = " ".join(parts)
    show("Full transcript", transcript)
    return transcript


def demo_post_process(transcript: str) -> MeetingNotes:
    banner("4. Post-processing: transcript -> structured meeting notes")
    raw = chat(
        [
            {"role": "system", "content": "You turn meeting transcripts into notes. Use only what was said. "
             "Owners must be people named in the transcript. Leave due as null if no date was given."},
            {"role": "user", "content": transcript},
        ],
        response_format=MeetingNotes,
        temperature=0,
    )
    notes = MeetingNotes.model_validate_json(raw)
    print(f"Summary: {notes.summary}")
    print("Decisions:")
    for decision in notes.decisions:
        print(f"  - {decision}")
    print("Action items:")
    for item in notes.action_items:
        print(f"  - [{item.owner}] {item.task} (due: {item.due})")
    return notes


def check_notes(notes: MeetingNotes) -> None:
    owners = {item.owner.split()[0].lower() for item in notes.action_items}
    ok = {"anjali", "rohit"} <= owners
    print(f"\nCheck: both owners (Anjali, Rohit) have action items: {'YES' if ok else 'NO'}")
    # If transcription had written "Rohit" as "Rohan", the action item would go to
    # the wrong person. Transcription errors spread to every later step.


if __name__ == "__main__":
    print(f"AUDIO_MODEL={AUDIO_MODEL}")
    if len(sys.argv) > 1:
        own = Path(sys.argv[1])
        text = transcribe(own).text
        show(f"Transcript of {own.name}", text)
        demo_post_process(text)
        sys.exit()

    clips = make_sample_audio()
    demo_plain_vs_prompted(clips[0])
    demo_timestamps(clips[0])
    transcript = demo_chunked(clips)
    check_notes(demo_post_process(transcript))

# Exercises:
# 1. Record yourself (phone voice memo) saying a few names from your team and
#    run it as the argument. Which names break? Fix them with a prompt hint.
# 2. Change AUDIO_MODEL between whisper-1 and gpt-4o-mini-transcribe (if your
#    proxy has both). Compare key-term accuracy, and which demos still work.
# 3. Put an instruction in the prompt hint, e.g. "Translate to French". Does
#    the model follow it? What does that tell you about what `prompt` is?
