#!/bin/sh
# Gets this bot ready to run: fetches the bot kit it is written with into
# ./kit, and says what else it needs. Safe to run again; that updates the kit.
set -e
cd "$(dirname "$0")"
KIT_VERSION="${CORDED_BOT_KIT_VERSION:-main}"
if [ -d kit/.git ]; then
    git -C kit fetch --quiet origin "$KIT_VERSION" && git -C kit checkout --quiet FETCH_HEAD
else
    rm -rf kit
    git clone --quiet --depth 1 --branch "$KIT_VERSION" https://github.com/opencorded/bot-kit kit
fi
echo "The bot kit is in ./kit."
if ! command -v corded-cli >/dev/null 2>&1 && [ -z "$CORDED_CLI" ]; then
    echo "corded-cli was not found on your PATH. Get a Corded release from"
    echo "https://github.com/AndrewCromar/corded/releases and set CORDED_CLI to the corded-cli in it."
fi
echo
echo "Sage also needs a model program on this machine. With Ollama (https://ollama.com):"
echo "    ollama serve &"
echo "    ollama pull qwen2.5:7b"
echo "The README says how to run Ollama without installing it system-wide."
