# Sage: a chat member run by a language model on your own machine

A member that a language model speaks through. The model runs on the same
machine as the bot, behind the chat API that Ollama, llama.cpp's server and LM
Studio all offer; nothing is sent anywhere else unless `--api` says so, and
the bot prints a warning at start if it does.

**Setting it up.** The model program is not part of this repository. With
Ollama (https://ollama.com), on the machine the bot will run on:

```sh
# 1. Ollama itself. Either the system install from their site, or, with no
#    install at all, their Linux archive unpacked into a folder of your own:
mkdir -p ~/ollama && cd ~/ollama
curl -fL https://ollama.com/download/ollama-linux-amd64.tar.zst | tar --zstd -x
#    (older releases are .tgz: curl -fL …/ollama-linux-amd64.tgz | tar -xz)

# 2. Run it, on this machine only, and fetch a model (about 4.7 GB):
OLLAMA_HOST=127.0.0.1:11434 ~/ollama/bin/ollama serve &
~/ollama/bin/ollama pull qwen2.5:7b

# 3. The bot:
python3 ai_bot.py 127.0.0.1:7443 --soul soul.example.txt --username sage
```

`OLLAMA_MODELS=/some/folder` puts the models somewhere other than
`~/.ollama`. Any other program that offers the same chat API (llama.cpp's
server, LM Studio) works with `--api`. The [host](https://github.com/opencorded/host) can start Ollama
and the bot together.

**Choosing the model from the app.** On Sage's page (members list, tap it,
Manage) the owner picks among the models Ollama has on that machine; the
change takes effect with the next answer. To have another model to pick,
fetch it on the machine: `ollama pull NAME`.

**What it will and will not say** is decided by two things: the soul file,
which is yours to write, and the model, which was trained with habits of its
own that no file fully overrides. `--model` takes any model your Ollama has;
people who want a model without those habits pick one made that way.

**When it answers.** It must answer a direct chat, a mention (`@sage`) and a
reply to one of its messages. It may answer, and decides for itself, in a
thread it has already spoken in, right after it answered someone (their next
message, within `--follow-up` seconds, 120), and in the channels chosen for
it. There the model is first asked one plain question, is this message meant
for the bot, and only a yes lets it speak: it joins in when it is asked or
talked about, and stays out when people are talking to each other. Alone with
one person in a channel, it takes everything as meant for it.

The owner, or anyone whose role has **Manage bots**, chooses a channel with
`!ai here` in it (or `!ai here #name` from anywhere); `!ai leave` undoes
that and `!ai where` lists them. One person gets at most `--per-minute`
answers a minute to mentions outside a direct chat (8 unless changed).

**How it answers.** The model decides each time, and the bot does what it
decided: a plain message, a reply attached to one message, a thread under
one, or only an emoji under it. Answers are short by instruction; a long one
goes into a thread and is cut at paragraph ends if it would not fit one
message. It shows as typing while the model works, and questions are answered
one at a time, so two people asking at once do not load the graphics card
twice.

It shows with a sprig of sage for a picture (`ai_bot.png`, drawn in
`ai_bot.svg`); `--display-name` and `--picture` change how it appears.

**Two files shape it**, both read again before every answer, so an edit shows
in the next one:

- `--soul FILE`: who it is. Its name, how it talks, what it will not do.
  `soul.example.txt` is a start. Without one it is a plain, friendly member.
- `--context FILE`: where it is and how the place works. `chat.md` beside the
  bot is used unless you give another: that this is a group chat, that
  answers are short, and when a reply, a thread or a reaction is the right
  form. Change the bot's manners there.

**Personalities.** It can be several characters and wears one at a time.
Each has its own display name, picture, character text and memories; what
one was told to remember the others do not know. You change them by talking
to it: "what personalities do you have", "here is a new one: a grumpy old
pirate captain called Barnacle", "switch to Barnacle", "from now on be more
sarcastic", "remember that I hate mornings", "make yourself a new profile
picture, a parrot in a storm" (drawn by the picture program, `--pictures`).
Or by command: `!persona` lists them, `!persona use NAME`,
`!persona new NAME: who it is`, `!persona show`, `!persona delete NAME`,
`!persona forget`. Switching and reshaping is for the owner and those with
Manage bots, unless `!persona open on` lets everyone; anyone can ask it to
remember. Each lives in the bot's folder as `personas/NAME/` with `soul.md`,
`memory.md`, `name.txt` and a picture, all of which can be edited by hand.
The first one is made from `--soul` on the first run.

How this works, because it matters with a small model: each message that
sounds like such a wish is first put to the model as a question of its own
(what, if anything, is being asked of the bot itself?), the bot carries the
answer out, and only then is the model asked to reply, told what was done.
Asked to act and to answer in one go, the model only said it had acted.

**What it keeps in mind.** The last `--memory` messages (30) of each chat, and
of each thread apart from its channel. Nothing extra is stored: after a
restart it reads them back from its own vault. `!forget` makes it start fresh
in the chat or thread it is said in. `!model` says which model it thinks with
and how long the last answer took.

**Which model.** Tried on one machine (RTX 4060, 8 GB) with the same scripted
chat, 2026-10-10:

| Model | Size on the card | An answer takes | How it did |
|---|---|---|---|
| `qwen2.5:7b` | about 5.4 GB | 0.3 to 1.5 s (the first after loading: 10 to 60 s) | Right answers, short, and the right form nearly every time: a reaction for thanks and good night, a reply to the older question it was asked about, a thread for the long one. The default |
| `llama3.2:3b` | about 3 GB | 0.2 to 1.4 s | Forms mostly right, but said it did not know the capital of Australia |
| `gemma3:4b` | about 4 GB | 0.3 to 1 s | Forms right (and a sleepy face for good night), but refused or fumbled easy questions |

A model this size gets facts wrong with a straight face now and then; the
context file tells it to say when it does not know, which helps and does not
cure it.

## Get it running

You need a Corded server to join, and `corded-cli`, the headless client that
comes with every Corded release ([get one here](https://github.com/AndrewCromar/corded/releases)).

```sh
git clone https://github.com/opencorded/bot-sage
cd bot-sage
./setup.sh              # fetches the bot kit into ./kit and says what else is needed
export CORDED_CLI=/path/to/corded-cli     # not needed if corded-cli is on your PATH
python3 ai_bot.py 127.0.0.1:7443 --soul soul.example.txt --username sage
```

The first run makes a folder holding the bot's keys and a passphrase file that
only your user can read. Keep it: it *is* the bot's account. On an invite-only
server give an invite link in place of the address. The bot can run on any
machine that reaches the server; it does not have to be the server's.

To keep it running, and to see it on a page with everything else, list it in
the [host](https://github.com/opencorded/host).

## Pictures

Sage makes no pictures itself. Asked for one ("draw me ...", "write a good
prompt and generate it"), it writes the description and hands it to the
[picture bot](https://github.com/opencorded/bot-pictures) on the same server
(`--pictures-bot NAME`, `pictures` unless said otherwise), for the person who
asked: that bot's own rules decide, and it posts the picture under their
message. Asked only for a prompt, Sage writes one out in words. It never types
commands, and stays out of what is said to another bot by name.

## Turning it off and on

`!NAME off` and `!NAME on`, where NAME is the bot's username, from the server's
owner or anyone whose role has **Manage bots**. Off, it stays connected but
shows as offline and hears nothing except the command that wakes it.
`!NAME status` says which it is. The choice is kept across restarts.

## What it can read

Whatever a member in its place could: every message in the channels and chats
it is in, from the moment it joined. Give it a role that sees only the
channels it needs.

## Tests

```sh
python3 tests/smoke.py /folder/with/cordedd/and/corded-cli
```

runs the bot against a throwaway server on this machine.

Part of [Corded](https://github.com/AndrewCromar/corded). Apache-2.0.
