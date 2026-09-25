#!/usr/bin/env bash
# Run any hands-on program, setting up the venv and .env on first use.
#
# Usage:
#   ./run.sh                                         # list programs
#   ./run.sh 01_prompt_engineering/01_instruction_context_query.py
#   ./run.sh 1                                       # same, by number (1-11)
#   ./run.sh 5 injection                             # extra args are passed through
set -euo pipefail

cd "$(dirname "$0")"

PROGRAMS=(
  01_prompt_engineering/01_instruction_context_query.py
  01_prompt_engineering/02_system_prompts_personas.py
  01_prompt_engineering/03_few_shot.py
  01_prompt_engineering/04_chain_of_thought.py
  01_prompt_engineering/05_failure_modes.py
  02_structured_outputs/06_json_mode_and_schema.py
  02_structured_outputs/07_function_calling.py
  02_structured_outputs/08_parse_and_validate.py
  03_multimodal/09_vision_inputs.py
  03_multimodal/10_audio_transcription.py
  03_multimodal/11_document_understanding.py
)

# 1. Virtual environment + dependencies
if [ ! -x .venv/bin/python ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
  .venv/bin/pip install -q -r requirements.txt
fi

# 2. .env with model + API key
if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env - set MODEL and your API key in it, then re-run."
  exit 1
fi

# Value of a variable: shell environment first, then .env (last match, quotes stripped)
env_value() {
  local from_shell="${!1:-}"
  if [ -n "$from_shell" ]; then
    echo "$from_shell"
  else
    grep -E "^$1=" .env | tail -1 | cut -d= -f2- | tr -d "\"'" || true
  fi
}

missing=()
require() {
  local value
  value="$(env_value "$1")"
  if [ -z "$value" ] || [[ "$value" == *"..."* || "$value" == *your-* || "$value" == *example.com* ]]; then
    missing+=("$1")
  fi
}

# Check the credentials the chosen provider needs (skipped for --offline runs)
if [[ " $* " != *" --offline "* ]]; then
  model="$(env_value MODEL)"
  model="${model:-gpt-4o-mini}"  # same default as common.py
  case "$model" in
    litellm_proxy/*) require LITELLM_PROXY_API_BASE; require LITELLM_PROXY_API_KEY ;;
    anthropic/*)     require ANTHROPIC_API_KEY ;;
    gemini/*)        require GEMINI_API_KEY ;;
    ollama/*)        ;;  # local, no key
    *)               require OPENAI_API_KEY ;;
  esac
  if [ ${#missing[@]} -gt 0 ]; then
    echo "MODEL=$model needs these set in .env (missing or still a placeholder):"
    printf '  %s\n' "${missing[@]}"
    exit 1
  fi
fi

# 3. Pick the program
if [ $# -eq 0 ]; then
  echo "Usage: ./run.sh <number|path> [args...]"
  for i in "${!PROGRAMS[@]}"; do
    echo "  $((i + 1))  ${PROGRAMS[$i]}"
  done
  exit 0
fi

target="$1"
shift
if [[ "$target" =~ ^[0-9]+$ ]]; then
  if [ "$target" -lt 1 ] || [ "$target" -gt "${#PROGRAMS[@]}" ]; then
    echo "No program number $target (choose 1-${#PROGRAMS[@]})."
    exit 1
  fi
  target="${PROGRAMS[$((target - 1))]}"
fi

exec .venv/bin/python "$target" "$@"
