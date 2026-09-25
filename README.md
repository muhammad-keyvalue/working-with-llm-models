# Working with LLM Models — Layer 2 Hands-on

Small, runnable Python programs for **Layer 2: Working with Models** of the AI engineering track.
Each program shows a naive approach first, then a better one, so you can see the difference in the output.

Every program talks to the model through [LiteLLM](https://docs.litellm.ai/), which gives you one
OpenAI-style API for any provider. Change `MODEL` in `.env` to switch providers.

## Quick start

Requires Python 3.10+.

```bash
./run.sh          # first run: creates .venv, installs dependencies, creates .env
# edit .env: set MODEL and your API key
./run.sh 1        # run the first program
```

## Configure `.env`

`.env` is created from `.env.example` and is git-ignored, so your key stays local.

```
MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-your-real-key
```

To use another provider, change `MODEL` and set that provider's key:

| Provider | `MODEL` example | Key variable |
|----------|-----------------|--------------|
| OpenAI | `gpt-4o-mini` | `OPENAI_API_KEY` |
| Anthropic | `anthropic/claude-sonnet-5` | `ANTHROPIC_API_KEY` |
| Google Gemini | `gemini/gemini-2.5-flash` | `GEMINI_API_KEY` |
| Local (Ollama) | `ollama/llama3.1` | none, just have Ollama running |

Any [model string LiteLLM supports](https://docs.litellm.ai/docs/providers) works. Use a model your account has access to.

## Running programs

```bash
./run.sh                 # list programs with their numbers
./run.sh 1               # run by number
./run.sh 01_prompt_engineering/01_instruction_context_query.py   # or by path
./run.sh 5 injection     # extra args are passed to the program
./run.sh 8 --offline     # runs without an API key
./run.sh 9 photo.png     # programs 9-11 accept your own image / audio / PDF
```

`run.sh` sets up `.venv` and `.env` if they're missing. It stops with a message while `.env` still has the placeholder key.

Without the script:

```bash
source .venv/bin/activate
python 01_prompt_engineering/01_instruction_context_query.py
```

## Programs (run in order)

### Part 1 — Prompt engineering

| # | Program | What it shows |
|---|---------|---------------|
| 1 | `01_prompt_engineering/01_instruction_context_query.py` | An unstructured prompt vs a delimited **Instruction → Context → Query** prompt |
| 2 | `01_prompt_engineering/02_system_prompts_personas.py` | The same question answered by 4 personas, and whether a persona keeps its rules over several turns |
| 3 | `01_prompt_engineering/03_few_shot.py` | Zero-shot vs few-shot ticket triage, scored against expected labels |
| 4 | `01_prompt_engineering/04_chain_of_thought.py` | Direct answer vs step-by-step reasoning on a billing calculation whose correct answer is computed in code |
| 5 | `01_prompt_engineering/05_failure_modes.py` | Hallucination, prompt injection and ambiguity: each one breaking first, then with a fix. Pass `hallucination`, `injection` or `ambiguity` to run one demo |

### Part 2 — Structured outputs & function calling

| # | Program | What it shows |
|---|---------|---------------|
| 6 | `02_structured_outputs/06_json_mode_and_schema.py` | Three levels: prompt-only JSON → JSON mode → schema enforcement with Pydantic |
| 7 | `02_structured_outputs/07_function_calling.py` | When the model chooses a tool, parallel calls, and the full tool loop with error handling |
| 8 | `02_structured_outputs/08_parse_and_validate.py` | Production parsing: extract → parse → validate (including business rules) → repair retry → fallback. `--offline` runs without an API key |

### Part 3 — Multimodal inputs

| # | Program | What it shows |
|---|---------|---------------|
| 9 | `03_multimodal/09_vision_inputs.py` | Image + text messages: vague vs precise questions, `detail` low vs high (accuracy vs tokens), several images in one request, and hallucination about things not in the image. Pass an image path to ask about your own |
| 10 | `03_multimodal/10_audio_transcription.py` | A transcription pipeline: plain vs vocabulary-hinted transcripts, timestamps, chunked long audio, then transcript → structured meeting notes. Pass an audio path to use your own recording |
| 11 | `03_multimodal/11_document_understanding.py` | Document AI: digital vs scanned PDFs, routing to text extraction or vision OCR, table extraction into a schema, business-rule validation, and sending the PDF file directly. Pass a PDF path to try your own |

Programs 9–11 generate their sample images, audio and PDFs into `03_multimodal/samples/` (git-ignored), so the correct answers are known and each result is scored.
They use three extra optional settings in `.env`:

| Variable | Default | Used for |
|----------|---------|----------|
| `VISION_MODEL` | same as `MODEL` | programs 9 and 11; must accept images |
| `AUDIO_MODEL` | `whisper-1` (`litellm_proxy/whisper-1` when `MODEL` uses the proxy) | program 10 transcription |
| `TTS_MODEL` | `tts-1` (`litellm_proxy/tts-1` when `MODEL` uses the proxy) | generating program 10's sample audio, once |

Each file ends with **Exercises**: small changes to try, with a question to answer about the result.

## Project layout

```
common.py                  shared chat()/ask() helpers; reads MODEL from .env
run.sh                     setup + runner script
01_prompt_engineering/     programs 1–5
02_structured_outputs/     programs 6–8
03_multimodal/             programs 9–11 (samples/ is generated)
requirements.txt           litellm, pydantic, python-dotenv, pillow, pypdf, fpdf2
.env.example               template for .env
```

## Troubleshooting

| Error | Fix |
|-------|-----|
| `AuthenticationError` | The key in `.env` is missing or wrong |
| `NotFoundError` / model not found | Change `MODEL` to a model your account can use |
| `ModuleNotFoundError: litellm` | Use `./run.sh` or `source .venv/bin/activate` first |
| `ModuleNotFoundError: PIL` / `pypdf` / `fpdf` | Existing `.venv` predates programs 9–11: run `.venv/bin/pip install -r requirements.txt` |
| Program 10 fails with model not found | Set `AUDIO_MODEL` / `TTS_MODEL` in `.env` to names your provider or proxy has |
| `RateLimitError` | Wait and retry, or use a model with higher limits |

## Notes

- Model output varies from run to run, even at `temperature=0`. Discuss the differences you see rather than expecting identical output.
- Not every model supports JSON-schema output or tool calling. `litellm.supports_response_schema(model)` and
  `litellm.supports_function_calling(model)` tell you which ones do.
