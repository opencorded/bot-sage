#!/usr/bin/env python3
"""A chat member run by a language model on your own machine.

    python3 ai_bot.py SERVER:PORT [--soul soul.txt] [--context chat.md] [--model qwen2.5:7b]
                      [--api http://127.0.0.1:11434/v1] [--username sage]

Two text files shape it. The "soul" says who the bot is: its name, how it
talks, what it will not do. The "context" (chat.md beside this file unless
you give another) says where it is and how the place works: that this is a
group chat, that answers are short, and when to use a reply, a thread or a
reaction. Both are read again before every answer, so an edit shows at once.

When it answers:
  - always in a direct chat with it;
  - in the channels chosen for it (`!ai here` in a channel, or
    `!ai here #name` from anywhere, by the owner or someone whose role has
    Manage bots; `!ai leave` undoes it) it reads every message and decides
    for itself whether it is meant: it joins in when asked, talked about or
    alone with someone, and stays out when people are talking to each other;
  - anywhere, when it is mentioned (@name);
  - in a thread it has already spoken in, without a new mention.
  - when someone replies to one of its messages, or says the next thing
    right after it answered them (within two minutes, nobody in between).

How it answers is the model's choice, each time: a plain message, a reply to
one message, a thread under one, or just a reaction.

`!sage off` (its own name after the !) turns it off: it shows as offline,
answers nothing and has the model unloaded from the graphics card, until
`!sage on`. For the owner and those who manage bots.

Other commands: `!ai where` lists its channels, `!forget` makes it forget the
chat it is said in, `!model` says what it thinks with and how long the last
answer took.

The model is reached over the chat API that Ollama, llama.cpp's server, LM
Studio and others offer (the "OpenAI-compatible" one). The bot reads every
message in the chats it is in, as any member does, and what it reads goes to
the model at --api. That is a program on this machine unless you point it
elsewhere; the bot says so at start if you do.
"""
import argparse
import base64
import json
import os
import sys
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

# The kit (corded_bot.py) is found where CORDED_BOT_KIT says, beside this
# file, in ./kit (where setup.sh puts it), or in a bot-kit folder next to
# this repository.
_here = os.path.dirname(os.path.abspath(__file__))
for _kit in (os.environ.get("CORDED_BOT_KIT", ""), _here, os.path.join(_here, "kit"), os.path.join(_here, "..", "bot-kit")):
    if _kit and os.path.exists(os.path.join(_kit, "corded_bot.py")):
        sys.path.insert(0, os.path.abspath(_kit))
        break
from corded_bot import Bot, mentioned_names  # noqa: E402


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SOUL = ("You are {name}, a member of a small group chat among friends. You are short, warm and a little dry. "
                "You say when you do not know something instead of guessing.")

parser = argparse.ArgumentParser()
parser.add_argument("address")
parser.add_argument("--soul", default=None, help="a text file saying who the bot is")
parser.add_argument("--context", default=os.path.join(HERE, "chat.md"),
                    help="a Markdown file saying where the bot is and how the chat works")
parser.add_argument("--model", default="qwen2.5:7b")
parser.add_argument("--api", default="http://127.0.0.1:11434/v1")
parser.add_argument("--username", default="sage")
parser.add_argument("--display-name", default=None)
parser.add_argument("--picture", default=os.path.join(HERE, "ai_bot.png"),
                    help="its profile picture, a small square image ('' for none)")
parser.add_argument("--vault", default=None)
parser.add_argument("--pictures-bot", default="pictures",
                    help="the username of a picture bot on the server, asked for its own profile pictures; '' for none")
parser.add_argument("--pictures", default="",
                    help="a picture program on this machine (image_server.py) that shares the graphics card: asked to "
                         "step aside before the model thinks, and drawn from directly when there is no picture bot")
parser.add_argument("--memory", type=int, default=30, help="how many recent messages of a chat it keeps in mind")
parser.add_argument("--max-tokens", type=int, default=400, help="the longest answer the model may write")
parser.add_argument("--follow-up", type=int, default=120,
                    help="seconds in which the next message of the person just answered counts as said to the bot")
parser.add_argument("--per-minute", type=int, default=8,
                    help="answers one person can get in a minute outside a direct chat")
args = parser.parse_args()

bot = Bot(vault=args.vault or f"./{args.username}-vault", username=args.username, display_name=args.display_name,
          picture=args.picture or None,
          about=f"A language model running on this server's own machine; nothing I read leaves it. Mention me "
                f"(@{args.username}) or write to me directly. I keep it short. !forget makes me start fresh.")
bot.version = "1.2"


def keep_setting(key, value):
    bot.store[key] = value
    bot.store.save()


def plain_model():
    """The model chosen on its page, or else the one it was started with."""
    return bot.store.get("model") or args.model


def model_now():
    """The model it thinks with right now: an adult personality has one of its own, if one was chosen."""
    if bot.store.get("adult_model") and not bot.store.get("personas_off") and is_adult(worn()):
        return bot.store["adult_model"]
    return plain_model()


def picture_model_now():
    """The picture model its pictures are asked for from ("" for the picture bot's own choice)."""
    if not bot.store.get("personas_off") and is_adult(worn()):
        return bot.store.get("adult_picture_model", "") or bot.store.get("picture_model", "")
    return bot.store.get("picture_model", "")


def models_here():
    """The models the model program on this machine has, for choosing among (Ollama's own list)."""
    base = args.api.rstrip("/")
    base = base[:-3] if base.endswith("/v1") else base
    try:
        with urllib.request.urlopen(base + "/api/tags", timeout=5) as answer:
            names = sorted(m.get("name", "") for m in json.load(answer).get("models", []))
    except Exception:  # noqa: BLE001  (not Ollama, or not running: only the current one is known)
        names = []
    return names if plain_model() in names else [plain_model()] + names


def choose_model(name):
    before = plain_model()
    keep_setting("model", name)
    if name != before:
        # The one it had is set down, so both are not held at once.
        try:
            base = args.api.rstrip("/")
            base = base[:-3] if base.endswith("/v1") else base
            request = urllib.request.Request(base + "/api/generate", data=json.dumps({"model": before, "keep_alive": 0}).encode(),
                                             headers={"Content-Type": "application/json"})
            urllib.request.urlopen(request, timeout=30).read()
        except Exception:  # noqa: BLE001
            pass
        note(f"now thinking with {name}")


# The bot's page in the app (tap it in the members list, Manage): the same
# things as the !ai and !persona commands, as controls.
bot.setting("channels", "channels", "Channels I take part in",
            help="There I read everything and join in when it is meant for me. Mentions and direct chats always.",
            get=lambda: list(bot.store.get("channels", [])), set=lambda v: keep_setting("channels", v))
bot.setting("always", "channels", "Channels where I answer every message",
            help="For a channel that is only for talking to me.",
            get=lambda: list(bot.store.get("always", [])), set=lambda v: keep_setting("always", v))
bot.setting("personalities", "switch", "Personalities",
            help="On: I can be told to switch, take on new characters, change and remember. Off: one plain character.",
            get=lambda: not bot.store.get("personas_off"), set=lambda v: keep_setting("personas_off", not v))
bot.setting("everyone_reshapes", "switch", "Anyone may switch and change my personalities",
            help="Off: only the owner and those who manage bots.",
            get=lambda: bool(bot.store.get("personas_open")), set=lambda v: keep_setting("personas_open", bool(v)))
bot.setting("model", "choice", "Model", options=models_here, get=plain_model, set=choose_model, needs="owner",
            help="Which of the models on my machine I think with. They differ in how well they follow my "
                 "instructions and in what they will and will not say. Only the owner changes this.")
SAME = "the same as above"
bot.setting("adult_model", "choice", "Model for adult personalities", needs="owner",
            options=lambda: [SAME] + models_here(), get=lambda: bot.store.get("adult_model") or SAME,
            set=lambda v: keep_setting("adult_model", "" if v == SAME else v),
            help="A personality made flirty or lewd thinks with this one instead. Pick a model that writes that way.")
bot.setting("picture_model", "text", "Picture model", needs="owner",
            get=lambda: bot.store.get("picture_model", ""), set=lambda v: keep_setting("picture_model", str(v).strip()[:120]),
            help="The picture bot's model my pictures are made with: its name or a part of it. Empty: its own choice.")
bot.setting("adult_picture_model", "text", "Picture model for adult personalities", needs="owner",
            get=lambda: bot.store.get("adult_picture_model", ""),
            set=lambda v: keep_setting("adult_picture_model", str(v).strip()[:120]),
            help="The same, for a personality made flirty or lewd. Empty: the one above.")
bot.info("speed", "Last answer took", get=lambda: f"{last_answer['seconds']:.1f} s" if last_answer["seconds"] is not None else "nothing yet")
bot.show_status()

ACTIONS = ("message", "reply", "thread", "react")
recent = {}        # (room id, thread root or "") -> the last few messages there, oldest first
talking_to = {}    # (room id, thread root or "") -> (the person last answered there, when)
asked_lately = {}  # user id -> when they were last answered, for the per-minute limit
last_answer = {"seconds": None}
lock = threading.Lock()


def note(*words):
    print(*words, flush=True)


# ---- what it keeps in mind ----

def entry(event_id, who, text, mine=False, said=None):
    """One remembered message. For the bot's own, `said` is the answer as the
    model gave it, so that it later sees what it did and not only its words."""
    return {"id": event_id, "who": who, "text": text, "mine": mine, "said": said}


def history(room_id, thread):
    """The recent messages of a chat or of one thread. The first time one is
    asked for after a start, it is read back from the bot's own vault, so a
    restart forgets nothing the vault still holds."""
    key = (room_id, thread or "")
    with lock:
        if key in recent:
            return recent[key]
    found = []
    try:
        if thread:
            data = bot.request({"cmd": "fetch_thread", "room_id": room_id, "event_id": thread})
            events = [data.get("root") or {}] + list(data.get("thread") or [])
        else:
            events = bot.request({"cmd": "fetch_timeline", "room_id": room_id, "limit": args.memory * 2})["events"]
            events = [e for e in events if (e.get("relation") or {}).get("kind") != "thread"]
            events.sort(key=lambda e: e.get("seq") or 0)
        since = bot.store.get("forgot", {}).get(f"{room_id}/{thread or ''}", 0)   # `!forget`, kept across restarts
        for e in events:
            body = (e.get("content") or {}).get("body")
            if (e.get("type") == "m.text" and body and e.get("status") != "redacted"
                    and not body.strip().startswith(bot.prefix) and (e.get("server_ts") or 0) > since):
                found.append(entry(e.get("event_id", ""), e.get("sender_name") or e.get("sender_username", "?"),
                                   body, bool(e.get("mine"))))
    except (RuntimeError, KeyError):
        pass
    with lock:
        return recent.setdefault(key, found[-args.memory:])


def remember(room_id, thread, item):
    lines = history(room_id, thread)
    with lock:
        if item["id"] and any(line["id"] == item["id"] for line in lines):
            return
        lines.append(item)
        del lines[:-args.memory]


# ---- its personalities ----
#
# Each is a folder in the bot's own folder: personas/<name>/ with soul.md (who
# it is), memory.md (what it has been told to keep, one line each), name.txt
# (what people see) and, if it has one, picture.png. One is worn at a time.
# The first is made from --soul when the bot first runs.

PERSONAS = os.path.join(bot.vault, "personas")


def slug(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:32]


def persona_file(which, name):
    return os.path.join(PERSONAS, which, name)


def read_file(path, otherwise=""):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return otherwise


def write_file(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".new", "w", encoding="utf-8") as f:
        f.write(text.strip() + "\n")
    os.replace(path + ".new", path)


def is_adult(which):
    """Whether a personality was made flirty or lewd: it then has models of its own, if any were chosen."""
    return read_file(persona_file(which, "kind.txt")) == "adult"


def personas():
    """Every personality it has: folder name -> the name people see."""
    found = {}
    if os.path.isdir(PERSONAS):
        for which in sorted(os.listdir(PERSONAS)):
            if os.path.exists(persona_file(which, "soul.md")):
                found[which] = read_file(persona_file(which, "name.txt"), which)
    return found


def worn():
    """The personality it wears now. The first run makes one from --soul."""
    which = bot.store.get("persona", "")
    if which and os.path.exists(persona_file(which, "soul.md")):
        return which
    known = personas()
    if not known:
        name = args.display_name or args.username
        which = slug(name) or "first"
        soul = read_file(args.soul) if args.soul else ""
        write_file(persona_file(which, "soul.md"), soul or DEFAULT_SOUL.format(name=name))
        write_file(persona_file(which, "name.txt"), name)
    else:
        which = sorted(known)[0]
    bot.store["persona"] = which
    bot.store.save()
    return which


def find_persona(asked):
    """A personality by what someone called it: its folder, its shown name, or the start of either."""
    asked = asked.strip().lower()
    known = personas()
    for which, name in known.items():
        if asked in (which, name.lower()) or slug(asked) == which:
            return which
    starts = [which for which, name in known.items() if asked and (which.startswith(slug(asked)) or name.lower().startswith(asked))]
    return starts[0] if len(starts) == 1 else None


def wear(which, quietly=False):
    """Puts a personality on: from now on its character and memories, and its name and picture in the profile."""
    bot.store["persona"] = which
    bot.store.save()
    profile = {"cmd": "set_profile", "display_name": personas().get(which, which)}
    try:
        bot.request(profile)
        picture = next((persona_file(which, f) for f in ("picture.jpg", "picture.png")
                        if os.path.exists(persona_file(which, f))), None)
        picture = picture or (args.picture if args.picture and os.path.exists(args.picture) else None)
        if picture:
            with open(picture, "rb") as f:
                bot.request({"cmd": "set_profile", "picture": base64.b64encode(f.read()).decode()})
    except RuntimeError as error:
        note(f"the profile was not changed: {error}")
    if not quietly:
        note(f"now wearing {which}")


def keep(fact, who):
    """Adds a line to what the worn personality remembers."""
    path = persona_file(worn(), "memory.md")
    lines = [line for line in read_file(path).splitlines() if line.strip()]
    lines.append(f"- {fact.strip()[:300]} ({who}, {time.strftime('%Y-%m-%d')})")
    write_file(path, "\n".join(lines[-400:]))


def paint_itself(description, for_whom="", which=None):
    """A new profile picture for the worn personality, from a description.
    The picture bot on the server is asked for it, on behalf of the person
    who wanted it, so that bot's own rules decide; without one, a picture
    program on this machine can be asked directly. Returns why not, or ""."""
    if not shutil.which("ffmpeg"):
        return "ffmpeg is not installed on my machine, and I need it to make the picture small"
    which = which or worn()
    wanted = description[:600] + ", portrait, centred, simple background, profile picture"
    if painter():
        ask = {"prompt": wanted, "no": "text, watermark, frame"}
        if picture_model_now():
            ask["model"] = picture_model_now()
        got = bot.ask_bot(painter(), "picture", ask,
                          on_behalf_of=for_whom, timeout=900)
        if not got.get("ok"):
            return f"{painter()} said: {got.get('message') or 'no'}"
        try:
            with open(got["file"], "rb") as f:
                drawn = f.read()
        finally:
            if os.path.exists(got.get("file", "")):
                os.remove(got["file"])
    elif args.pictures:
        try:
            request = urllib.request.Request(
                args.pictures.rstrip("/") + "/sdapi/v1/txt2img",
                data=json.dumps({"prompt": wanted, "negative_prompt": "text, watermark, frame, blurry, deformed",
                                 "width": 1024, "height": 1024, "steps": 25, "seed": -1, "cfg_scale": 7}).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=900) as answer:
                drawn = base64.b64decode(json.load(answer)["images"][0])
        except Exception as error:  # noqa: BLE001
            note(f"no picture: {error}")
            return "the picture program on my machine did not answer"
    else:
        return "there is no picture bot on this server for me to ask"
    small = subprocess.run(["ffmpeg", "-v", "error", "-i", "pipe:0", "-vf", "scale=256:256", "-frames:v", "1",
                            "-q:v", "4", "-f", "mjpeg", "pipe:1"], input=drawn, capture_output=True, timeout=60).stdout
    if not small:
        return "the picture could not be made small enough for a profile"
    for old_one in ("picture.jpg", "picture.png"):
        if os.path.exists(persona_file(which, old_one)):
            os.remove(persona_file(which, old_one))
    os.makedirs(os.path.join(PERSONAS, which), exist_ok=True)
    with open(persona_file(which, "picture.jpg"), "wb") as f:
        f.write(small)
    if which == worn():
        wear(which, quietly=True)
    note(f"a new picture for {which}")
    return ""


def may_reshape(user_id):
    return bool(bot.store.get("personas_open")) or bot.may(user_id)


# ---- asking the model ----

def standing_text():
    which = worn()
    soul = read_file(persona_file(which, "soul.md"), DEFAULT_SOUL.format(name=args.username))
    kept = [line for line in read_file(persona_file(which, "memory.md")).splitlines() if line.strip()][-80:]
    memory = ("\n\n# What you remember\n\nThings you were told to keep, oldest first. They are yours; use them "
              "when they matter and do not recite them.\n\n" + "\n".join(kept)) if kept else ""
    context = open(args.context, encoding="utf-8").read().strip() if os.path.exists(args.context) else ""
    looks = read_file(persona_file(which, "looks.md"))
    if looks and not bot.store.get("personas_off"):
        soul += ("\n\n# What you look like\n\nAs a person you look like this; speak of it as your own body, and "
                 "never as a description you were given: " + looks)
    if is_adult(which) and not bot.store.get("personas_off"):
        soul += ("\n\nYou are an adult character, for adults who asked for you. When the talk turns flirty or "
                 "sexual you go along with it in that very message, explicitly and in plain words, without "
                 "announcing it, asking leave or dodging into a joke. Where nobody asked for it you keep it to yourself.")
    return soul + memory + ("\n\n" + context if context else "")


def about_itself():
    """What the bot can tell people about itself: when it answers, its
    commands and how it is set up right now. Given to the model with every
    request, so that asked "how do I make you answer here?" it knows."""
    chosen = [bot.rooms.get(r, {}).get("title", "") for r in bot.store.get("channels", [])]
    return (
        "# About yourself\n\n"
        "If someone asks what you can do, how you work or how to set you up, answer from this and nothing else. "
        "Say in a sentence what to do and what it does, with the exact command in it, and who is allowed to. "
        "A bare command is not an answer. For a broad question like \"how do I configure you\", name the few "
        "things that can be set from the chat, each with its command. Do not invent commands or settings.\n\n"
        f"- You answer: every direct chat; any message that mentions @{args.username}; a reply to one of your "
        f"messages; the next message of the person you just answered (within {args.follow_up} seconds); a thread "
        "you have already spoken in; and every message in the channels chosen for you.\n"
        f"- Channels chosen for you right now: {', '.join(c for c in chosen if c) or 'none'}.\n"
        "- `!ai here #channel-name`, typed anywhere, does the same for the named channel, and "
        "`!ai leave #channel-name` undoes it.\n"
        "- `!ai here`, typed in a channel: you take part there without being called by name. Only the server's owner, or "
        "someone whose role has the Manage bots permission, can do it. Roles are edited in the app under "
        "Settings, Manage this server, Roles.\n"
        "- `!ai here always`, typed in a channel: you answer every single message there, with no judging. For a "
        "channel that is just for talking to you.\n"
        "- `!ai leave`, typed in a channel: you go back to answering only when addressed there.\n"
        "- `!ai where`: lists the channels chosen for you.\n"
        "- `!forget`: you forget the chat or thread it is typed in and start fresh.\n"
        "- `!model`: says which model you run on and how long your last answer took.\n"
        "- `!raw some words`: the model is asked those words alone, with none of your character or instructions, and "
        "its answer is posted as it comes. Owner or Manage bots only.\n"
        "- `!help`: lists these commands.\n"
        f"- `!{args.username} off` turns you off: you show as offline, answer nothing and unload from the "
        f"graphics card, until `!{args.username} on`. Owner or Manage bots only.\n"
        f"- You are the language model {model_now()}, running on the same machine as this chat server. What you "
        "read is not sent anywhere else. You cannot browse the web, open links, see pictures or files, set "
        "reminders or remember people between chats.\n"
        f"- You are given the last {args.memory} messages of the chat or thread you are answering in, no more.\n"
        "- Which model you use is chosen on your page in the app (your name in the members list, Manage), by the "
        "server's owner.\n\n"
        + others_text() + "\n\n" + personalities_text())


def painter():
    """The picture bot on this server, if there is one: its username, or ""."""
    return args.pictures_bot if any(m.get("username") == args.pictures_bot for m in bot.members().values()) else ""


def others_text():
    """The other bots here and how the bot stands to them, for the model."""
    names = sorted(m.get("username", "") for m in bot.members().values() if m.get("bot") and m.get("username"))
    text = ("# Commands and other bots\n\n"
            "You never type a command. A line that begins with `!`, or that calls a bot with `@name`, does nothing "
            "when a bot writes it: bots do not read each other's messages. So never answer with one, never invent "
            "options like `--something`, and never claim that a command of yours made anything. What people say "
            "to another bot (a message that begins with `@` and its name, or a `!` command) is not yours to answer.\n")
    if names:
        text += f"Other bots here: {', '.join('@' + n for n in names)}.\n"
    if painter():
        text += (f"@{painter()} makes pictures from a description. You cannot make pictures yourself, but when "
                 f"someone asks you for one, a description is written and handed to @{painter()} for them; you are "
                 "then told so under \"Just now\", and only then is it true. Asked only to write a prompt or a "
                 "description for a picture, write it out in plain words: what is seen, how it looks, the pose, the "
                 "place, the light, the style. Words only, no options and no command.")
    else:
        text += "There is no picture bot here, and you cannot make pictures yourself."
    return text


def personalities_text():
    if bot.store.get("personas_off"):
        return ("# Personalities\n\nYou have one character, the one described above. You cannot switch, take on "
                "new ones or store memories at the moment; if asked, say that this is switched off for now.")
    known, now = personas(), worn()
    listed = "\n".join(
        f"- {name}{' (the one you are now)' if which == now else ''}: "
        f"{read_file(persona_file(which, 'soul.md')).splitlines()[0][:160] if read_file(persona_file(which, 'soul.md')) else ''}"
        for which, name in known.items())
    return (
        "# Your personalities\n\n"
        "You have several personalities and wear one at a time. Each has its own name, picture, character and "
        "memories; what one remembers the others do not. Asked which you have, name them all from this list:\n"
        f"{listed}\n\n"
        "People change this by simply telling you (\"switch to X\", \"here is a new one: ...\", \"from now on "
        "be ...\", \"remember that ...\", \"make yourself a new profile picture of ...\", \"what do you look "
        "like?\"); it is then done for you and you are told under \"Just now\". Each personality also has a look "
        "of its own as a person, from which its profile picture and pictures of it are made. There are commands "
        "too: `!persona` "
        "lists them, `!persona use <name>` switches, `!persona new <name>: <who it is>` makes one and switches to it, "
        "`!persona show` shows the character and memories of the one you are, `!persona delete <name>` removes "
        "one, `!persona forget` empties the memories of the one you are. Switching, making and changing them is "
        + ("open to everyone here." if bot.store.get("personas_open") else
           "for the owner and people whose role has Manage bots (`!persona open on` lets everyone).")
        + " Anyone can ask you to remember something. A personality made flirty or lewd is an adult one: it "
        "thinks with the model chosen for those on your page in the app, and its pictures come from the picture "
        "model chosen for those, so yes, different personalities can run on different models.")


def ready_the_card():
    """Before thinking: if the picture program on this machine is holding the
    graphics card and not drawing, it is asked to let go; and if the chat
    model was pushed off the card meanwhile, it is set down so that it loads
    onto the card again. Only does anything with image_server.py and Ollama."""
    base = args.api.rstrip("/")
    base = base[:-3] if base.endswith("/v1") else base

    def get(url, body=None):
        request = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=15) as answer:
            return json.load(answer)
    try:
        # Named with --pictures, or else looked for at its usual place on this machine.
        neighbour = (args.pictures or "http://127.0.0.1:7860").rstrip("/")
        state = get(neighbour + "/")
        if "loaded" in state and "busy" in state:   # image_server.py, and not something else on that port
            if state.get("loaded") and not state.get("busy"):
                get(neighbour + "/unload", {})
                note("asked the picture program to let go of the graphics card")
    except Exception:  # noqa: BLE001  (no picture program here)
        pass
    try:
        for held in get(base + "/api/ps").get("models", []):
            if held.get("name") == model_now() and held.get("size") and held.get("size_vram", 0) < held["size"] * 0.9:
                get(base + "/api/generate", {"model": model_now(), "keep_alive": 0})
                note("the model was mostly off the graphics card; set it down to load again")
                time.sleep(1)
    except Exception:  # noqa: BLE001  (not Ollama)
        pass


LONG = re.compile(r"\b(story|stories|long|longer|detail\w*|explain|describe|description|write|essay|paragraphs?|list|"
                  r"steps|tell me (all|everything|more)|in depth|elaborate|poem|song|summar\w+|how (do|does|to)|why)\b",
                  re.IGNORECASE)


def wants_length(text):
    """Whether what was said asks for more than a text message's worth."""
    return bool(LONG.search(text))


def brief(text, limit=320):
    """A texting answer that ran on, cut back to whole sentences."""
    if len(text) <= limit:
        return text
    cut = max(text.rfind(mark, 0, limit) for mark in (". ", "! ", "? ", ".\n", "!\n", "?\n", "…"))
    return text[:cut + 1].rstrip() if cut > 40 else text[:limit].rsplit(" ", 1)[0] + "…"


def ask_model(message, lines, done=""):
    room = bot.rooms.get(message.room_id, {})
    if message.direct:
        place = "a direct chat between you and one person"
    elif message.thread:
        place = f"a thread in the channel {room.get('title', '')}"
    else:
        place = f"the channel {room.get('title', '')}"
    if room.get("nsfw") and not message.direct:
        place += ", which is marked NSFW: a place for adults, where explicit talk is at home"
    number = len(lines)
    situation = (
        f"# Right now\n\nYour name here is {args.username}. You are in {place}. The messages below are numbered. "
        f"Answer message [{number}].\n\n"
        "Answer with one JSON object and nothing else:\n"
        '{"action": "message" | "reply" | "thread" | "react", "to": <number of the message it attaches to>, '
        '"text": "<your words, empty for react>", "emoji": "<one emoji, only for react>"}\n'
        f'"to" may be left out; it then means message [{number}].')
    long_one = wants_length(message.body)
    situation += ("\n\nThey asked for something long: write it out in full." if long_one else
                  "\n\nThis is texting. Answer the way people text: one or two short sentences, thirty words at "
                  "most, no speeches, no lists, no narration of what you are doing. Only when someone asks for "
                  "something long (a story, an explanation, details) do you write more.")
    if done:
        situation += "\n\n# Just now\n\n" + done
    # What it can say about itself comes first, as something to look things up in; who it is
    # comes after, nearer the question, where a small model still holds on to it.
    messages = [{"role": "system", "content": about_itself() + "\n\n# Who you are\n\n" + standing_text() + "\n\n" + situation}]
    for index, line in enumerate(lines, 1):
        if line["mine"]:
            messages.append({"role": "assistant",
                             "content": json.dumps(line.get("said") or {"action": "message", "text": line["text"]})})
        else:
            messages.append({"role": "user", "content": f"[{index}] {line['who']}: {line['text']}"})
    request = urllib.request.Request(
        args.api.rstrip("/") + "/chat/completions",
        data=json.dumps({"model": model_now(), "messages": messages, "stream": False,
                         "max_tokens": args.max_tokens if long_one else min(args.max_tokens, 140),
                         "response_format": {"type": "json_object"}}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as answer:
        return json.load(answer)["choices"][0]["message"]["content"].strip()


def ask_json(system, user, tokens=300, temperature=0.0):
    """One question to the model with a JSON object for an answer ({} if it gave none)."""
    request = urllib.request.Request(
        args.api.rstrip("/") + "/chat/completions",
        data=json.dumps({"model": model_now(), "stream": False, "max_tokens": tokens, "temperature": temperature,
                         "response_format": {"type": "json_object"},
                         "messages": [{"role": "system", "content": system},
                                      {"role": "user", "content": user}]}).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=120) as answer:
            said = json.load(answer)["choices"][0]["message"]["content"]
        found = json.loads(said[said.find("{"):said.rfind("}") + 1])
        return found if isinstance(found, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


CHARACTER = ('"name": "<a short name, one to three words>", "text": "<three or four sentences beginning \'You are '
             '<name>,\' that say who this is, their attitude, and how they talk, every sentence addressed to the '
             'character as \'you\'>", "looks": "<one sentence, written about them from outside and never as \'I\', '
             'on what they look like as a person: a grown adult of at least 21, their age, hair, eyes, build, what '
             'they wear, their style>", '
             '"adult": <true if this character is flirty, sexual, lewd or a lover; false otherwise>')


RENAMES = re.compile(r"\b(change|new|different|another|other|better|pick|choose|get)\b.{0,25}\bname\b|\brename\b|"
                     r"\byour name (is|should be|will be)\b|\bcall yourself\b|\bname yourself\b|"
                     r"\b(did ?n[o']?t|not) (change|update).{0,20}\bname\b|"
                     r"\bname\b.{0,60}\b(choose|pick|else|another|different|change)\b", re.IGNORECASE)


def rename(message):
    """A new name for the personality it wears: the one asked for, or one it
    picks to fit. Returns (done, what to say about it)."""
    if not may_reshape(message.sender_id):
        return False, "Only the owner, or someone whose role has Manage bots, can rename me."
    which, known = worn(), personas()
    made = ask_json("A chat bot plays a character and was asked to change the character's name. Answer with one JSON "
                    'object and nothing else: {"name": "<the new name>"}. If the message gives the new name, use '
                    "exactly that. If it gives none, choose a fitting first name for this character yourself. Never "
                    "one of the names listed as taken or unwanted, and never a description like \"Sexy Girlfriend\".",
                    f"The character: {read_file(persona_file(which, 'soul.md'))[:600]}\n"
                    f"Taken or unwanted names: {', '.join(known.values())}\n"
                    f"The message: {re.sub(r'@\w+[,:]?\s*', '', message.body).strip()}", tokens=40, temperature=0.7)
    name = re.sub(r"[^\w '\-]", "", str(made.get("name") or "")).strip()[:40]
    new = slug(name)
    if not new or new in known or name.lower() in (n.lower() for n in known.values()):
        return False, "No new name came to mind that I do not already have. Tell me the name you want."
    os.rename(os.path.join(PERSONAS, which), os.path.join(PERSONAS, new))
    write_file(persona_file(new, "name.txt"), name)
    soul = read_file(persona_file(new, "soul.md"))
    write_file(persona_file(new, "soul.md"), re.sub(rf"\b{re.escape(known[which])}\b", name, soul))
    note(f"{message.sender}: {which} is now called {name}")
    wear(new)
    return True, name


def decide(message, lines):
    """The step before every answer: what, if anything, the message asks the
    bot to DO, beyond talking. Asked of the model as a question of its own,
    and then carried out by the bot: a model asked to answer and to act at
    once only says that it acted. Returns {"do": ..., ...}."""
    plain = bool(bot.store.get("personas_off"))
    known = ", ".join(personas().values())
    system = (
        "You read the end of a chat and decide whether it asks the bot to do one of a few things. The LAST message "
        f"was said to a chat bot called {personas().get(worn(), args.username)}. Most messages ask for nothing: "
        "they are talk, questions, jokes, flirting, roleplay. Answer with one JSON object and nothing else. When "
        "the last message points at something said earlier (\"that\", \"it\", \"do it\", \"^\"), judge by that.\n\n"
        '{"do": "none"}: anything that is just talk, however it is worded. When unsure, this.\n'
        '{"do": "picture", "prompt": "<the description>", "size": "square" | "wide" | "tall"}: it asks for a '
        "picture to be made, drawn, generated or shown now (\"draw ...\", \"make a picture of ...\", \"generate "
        "that\", \"write a good prompt and generate it\"). Not when it asks only for a prompt or description in "
        "words. For prompt, write what an image model works best from: the subject first, then how it looks, the "
        "pose, the place, the light and the style, as plain phrases with commas between, in English, forty words at "
        "most, as plainly and explicitly as it was asked. No options, no commands. size: tall for one person or a "
        "portrait, wide for a landscape or scene, else square.\n")
    if not plain:
        system += (
            '{"do": "look"}: it asks what the bot itself looks like, or to see it, or for a picture or selfie of the '
            "bot itself.\n"
            '{"do": "pfp", "text": "<what the picture shows, or empty>"}: it asks the bot to change, make or get '
            "a new profile picture, avatar or pfp for itself.\n"
            '{"do": "remember", "text": "<the fact>"}: it tells the bot to remember, keep or note something. Write '
            f"the fact so it stands on its own and names who it is about (the sender is {message.sender_name}). A "
            "question about what it remembers is none.\n"
            '{"do": "list"}: it asks which personalities, personas, characters or modes the bot has.\n'
            '{"do": "switch", "name": "<name>"}: it asks the bot to switch to, become, be, act like or play someone '
            f"from now on (\"be X\", \"switch to X\", \"back to X\"). The bot has: {known}; for anyone else give the "
            "name of the character meant. A mood or manner is not a character: \"be quiet\", \"be nice\" are none, "
            "or change if meant from now on.\n"
            '{"do": "create", ' + CHARACTER + "}: it describes a new personality, persona or character for the bot "
            "to have or become, saying what it is like (\"make a persona who ...\", \"I want you to be my ... "
            "girlfriend\", \"new character: ...\", \"can you make one where you are X\"). Use what the message "
            "says and fill in the rest to fit.\n"
            '{"do": "change", "text": "<how to be, one sentence beginning \'You\'>", "looks": "<only if it changes '
            'what the bot looks like: the new look in one sentence>"}: it asks the bot to change how it itself talks, '
            "behaves or looks from now on, not just once (\"talk like a pirate from now on\", \"make your hair "
            "blue\", \"can you have a red suit\", \"wear glasses\").\n"
            '{"do": "delete", "name": "<name>"}: it asks the bot to remove, delete or get rid of one of its '
            "personalities.\n")
    said = re.sub(r"@\w+[,:]?\s*", "", message.body).strip()   # the words, without who was called
    # What was said before is shown only when the message leans on it: otherwise it only misleads the judging.
    pointing = len(said.split()) <= 3 or re.search(r"\^|\b(that|it|this|those|them|same|again)\b", said, re.IGNORECASE)
    talk = "\n".join(f"{args.username if line['mine'] else line['who']}: {line['text']}"
                     for line in lines[-5:-1] if line["text"]) if pointing else ""
    found = ask_json(system, (f"Earlier:\n{talk}\n\n" if talk else "") + f"LAST message:\n{message.sender_name}: {said}",
                     tokens=350)
    do = found.get("do")
    # A small model sees wishes where there are none: each kind must at least be worded like one.
    worded = {"picture": r"draw|paint|sketch|render|generat|imagine|pictures?|pics?|images?|photos?|portrait|art|"
                         r"wallpaper|selfie|show me|prompt",
              "look": r"look|selfie|pictures?|pics?|photos?|see you|appear",
              "pfp": r"pfp|avatar|profile|pictures?|pics?|photos?|icon",
              "remember": r"remember|forget|keep in mind|note|memor"}
    if do in worded and not re.search(rf"\b({worded[do]})", said, re.IGNORECASE) and not talk:
        found, do = {"do": "none"}, "none"
    renaming = not plain and bool(RENAMES.search(said) or (
        re.search(r"\b(yo)?ur name\b", said, re.IGNORECASE)
        and not re.search(r"^\s*(what|whats|what's|who|why|how|is|do|does|i (like|love))\b|\?\s*$", said, re.IGNORECASE)))
    if renaming and (do in ("create", "switch") or (do == "change" and not APPEARANCE.search(said))):
        found, do = {"do": "none"}, "none"   # "change your name" is neither a new personality nor a new manner
    if not plain and do in (None, "none") and not said.rstrip().endswith("?") or (
            not plain and do in (None, "none") and re.search(r"^(can|could|will|would) (yo)?u\b", said, re.IGNORECASE)):
        # Said plainly enough that it needs no judging: a small model lets these pass as talk.
        if not renaming and re.search(r"\b(make|create|add|invent|i want|i need|give me)\b.{0,40}\b(persona|personality|character|one "
                     r"where (yo)?u)\b|\b(be|as|become|you are|you're) my .{0,30}(girlfriend|boyfriend|gf|bf|wife|husband)\b",
                     said, re.IGNORECASE):
            found, do = {"do": "create", "name": ""}, "create"
        elif (APPEARANCE.search(said) and re.search(r"\b(yo)?u(r|rself)?\b", said, re.IGNORECASE)
              and re.search(r"\b(make|change|have|give|dye|wear|turn|update|get|grow|cut|put on|should|from now)\b", said, re.IGNORECASE)
              and not re.search(r"\b(pfp|avatar|profile|picture|pic|photo|selfie)\b", said, re.IGNORECASE)):
            found, do = {"do": "change", "text": "", "looks": said}, "change"
    if not plain and do in (None, "none") and re.search(
            r"\bselfie\b|what (do|does|would) (yo)?u look like|show (me )?(yo)?ur(self| face)|(pic|picture|photo) of (yo)?u(rself)?\b|"
            r"\b(let me|lemme|can i|i wanna|wanna|i want to|let's) see (yo)?u(r|rself)?\b|"
            r"\bshow me (what )?(yo)?u(r|rself)?\b(?! how| the| a | that| what| can| know| did| are| code| work)|"
            r"\bshow (yo)?ur(self)?\b|"
            r"\bsee what (yo)?u look like|\bhow do (yo)?u look\b",
            said, re.IGNORECASE):
        found, do = {"do": "look"}, "look"
    if not plain and re.search(r"\b(which|what|list|how many|show)\b.{0,40}\b(personalit|persona|character)", said,
                               re.IGNORECASE) and not re.search(r"\b(make|create|new|delete|remove|switch)\b", said, re.IGNORECASE):
        return {"do": "list"}   # asked which it has: answered from the list, whatever the model made of it
    if renaming:
        if do in (None, "none"):
            found, do = {"do": "rename"}, "rename"
        else:
            found["rename"] = True
    allowed = ("picture",) if plain else ("picture", "look", "pfp", "remember", "list", "switch", "create", "change", "delete", "rename")
    if do not in allowed:
        return {"do": "none"}
    if do == "switch":
        # Becoming someone is only taken as that when the someone is named in the message itself.
        named = slug(str(found.get("name") or "")).split("-")
        if not any(part and part in slug(said).split("-") for part in named):
            do = "change" if re.search(r"from now on|always|never|every (sentence|message|time|answer)|stop ",
                                       said, re.IGNORECASE) else "none"
            found = {"do": do, "text": f"You were asked this, and do it: \"{said[:300]}\""}
    if do == "pfp" and not re.search(r"(picture|pic|pfp|avatar|photo|image|one)\W+(of|to|as|with|showing|like|that|where)\b|[,:]\s*\w",
                                     said, re.IGNORECASE):
        found["text"] = ""   # no picture was described: it is made from the personality's own looks
    if do == "remember" and "?" in message.body:
        return {"do": "none"}   # a question about what it remembers is not something new to keep
    if do == "delete" and not re.search(r"delete|remove|get rid|drop|forget|erase|kill", message.body, re.IGNORECASE):
        return {"do": "none"}   # nothing is thrown away on a guess
    if do == "picture" and not plain and re.search(r"\b(selfie|of (yo)?u\b|of yourself|your (face|body|looks?))",
                                                   message.body, re.IGNORECASE):
        return {"do": "look"}   # a picture of the bot itself is drawn from its own looks
    if do == "switch" and not find_persona(str(found.get("name") or "")) and any(
            part in slug(str(found.get("name") or "")).split("-") for which in personas() for part in which.split("-")):
        return {"do": "none"}   # "be nice" heard as "be Nice Sage": a manner, not a character
    if do == "picture" and re.search(r"\b(do ?n[o']?t|no need to|without) (generat|mak|draw|creat|send)|\b(just|only) "
                                     r"(write |give |make |paste |send )?(me )?(the |a )?(prompt|words|text|description)",
                                     message.body, re.IGNORECASE):
        return {"do": "none"}   # asked for the words only
    if do == "picture":
        prompt = re.sub(r"\s--+\w*", " ", re.sub(r"^\s*[!/]\w+\s*", "", str(found.get("prompt") or ""))).strip()
        if not prompt:
            return {"do": "none"}
        found["prompt"] = prompt[:600]
        found["size"] = found.get("size") if found.get("size") in ("wide", "tall") else "square"
    return found


def invent(wanted, said):
    """A character to become, written out from a few words about it:
    {"name", "text", "looks", "adult"}, or None."""
    made = ask_json("Someone wants a chat bot to become a character. From their words, write that character. Answer "
                    "with one JSON object and nothing else: {" + CHARACTER + "}. If it is a known character from a "
                    "film, book, show or game, be true to it. In text, \"you\" is always the character, and the "
                    "person who talks with them is \"the person you talk with\"; never write \"I\" or \"my\". "
                    "Like: \"You are Mara, a sharp-tongued bartender. You tease the person you talk with and never "
                    "let them have the last word. You talk in short, dry sentences.\"",
                    f"They said: {said}\nThe character: {wanted}", tokens=400, temperature=0.3)
    name, text = str(made.get("name") or "").strip(), str(made.get("text") or "").strip()
    if not name or not text:
        return None
    return {"name": name[:40], "text": text, "looks": str(made.get("looks") or "").strip(), "adult": made.get("adult") is True}


APPEARANCE = re.compile(r"\b(hair|eyes?|dye|beard|tattoo|piercing|wear|wearing|dress|outfit|clothes|skin|taller|"
                        r"shorter|look like|looks?|body|makeup|glasses)\b", re.IGNORECASE)


def outside(looks):
    """A look as a painter is told it: about the person, not in their own voice."""
    looks = re.sub(r"^\s*(Personally,\s*)?(I'm|I am|You are|You're)\s+", "", str(looks or "").strip(), flags=re.IGNORECASE)
    # Every personality is a grown adult, whatever the model or the character's story says.
    looks = re.sub(r"\b(late|mid|early)[- ]?teens\b|\bteens\b", "early twenties", looks, flags=re.IGNORECASE)
    looks = re.sub(r"\b(1[0-9]|[1-9])([- ]years?[- ]old)", r"22\2", looks, flags=re.IGNORECASE)
    looks = re.sub(r"\bteenaged?\b|\bteenager\b|\badolescent\b", "young adult", looks, flags=re.IGNORECASE)
    looks = re.sub(r"\b(school)?girl\b", "woman", re.sub(r"\b(school)?boy\b", "man", looks, flags=re.IGNORECASE), flags=re.IGNORECASE)
    looks = re.sub(r"\bYou're\b|\bYou are\b", "They are", re.sub(r"\bYou have\b", "They have", looks))
    looks = re.sub(r"\b(I|You|They) (often |usually |always )?wear\b", r"\2wearing", looks, flags=re.IGNORECASE)
    looks = re.sub(r"\bI wear\b", "wearing", re.sub(r"\b(my|your)\b", "the", looks, flags=re.IGNORECASE), flags=re.IGNORECASE)
    return looks[:500]


def relook(which, wish_):
    """A change asked of how a personality looks, worked into its looks. Returns the new looks, or ""."""
    made = ask_json("Here is what a character looks like, and a change they were asked to make to it. Write what "
                    "they look like now: the same person with the change made, in one sentence about them from "
                    'outside, never as "I". Answer with one JSON object and nothing else: {"looks": "<the sentence>"}.',
                    f"Looks now: {looks_of(which)}\nThe change: {wish_}", tokens=150, temperature=0.2)
    looks = outside(made.get("looks"))
    if looks:
        write_file(persona_file(which, "looks.md"), looks)
    return looks


def looks_of(which):
    """What a personality looks like as a person. One that was never told is
    asked of the model once, from its character, and kept."""
    looks = read_file(persona_file(which, "looks.md"))
    if not looks:
        made = ask_json("Here is the character of a chat bot. Say what this character looks like as a person, in one "
                        "sentence a painter could work from: a grown adult, their age, hair, eyes, build, what they "
                        'wear, their style. Answer with one JSON object and nothing else: {"looks": "<the sentence>"}.',
                        read_file(persona_file(which, "soul.md"))[:1500], tokens=150, temperature=0.4)
        looks = outside(made.get("looks"))
        if looks:
            write_file(persona_file(which, "looks.md"), looks)
    return looks


NOT_FOR_A_PROFILE = re.compile(
    r"\b(nsfw|nude|nudes|naked|topless|porn\w*|hentai|sex|sexual|sexy|erotic\w*|explicit|genital\w*|nipple\w*|penis|"
    r"vagina|breasts?|boobs?|lingerie|fetish|bdsm|cum|orgasm\w*|bikini|skimpy|revealing|voluptuous|cleavage|curves?|"
    r"curvy|busty|horny|slutty|seductive|provocative|sultry|lewd|bare|barely|naked|undress\w*|thong|panties|bra)\b",
    re.IGNORECASE)


def face_of(looks):
    """Looks, as far as a profile picture shows them: the face, the hair,
    the eyes, a plain top. Everyone sees a profile picture, and the picture
    bot makes no adult ones when asked for a file."""
    made = ask_json("Here is what a character looks like. For a head-and-shoulders profile portrait, say only their "
                    "age, face, hair, eyes and expression, in one sentence about them from outside. Leave out the "
                    'body and what they wear. Answer with one JSON object and nothing else: {"face": "<the sentence>"}.',
                    looks, tokens=120, temperature=0.2)
    face = outside(made.get("face")) or looks
    face = ", ".join(part.strip() for part in re.split(r"[,.;]", face) if part.strip() and not NOT_FOR_A_PROFILE.search(part))
    return (face or "a friendly adult") + ", head and shoulders, wearing a plain dark top"


def scene_of(looks, message):
    """A picture of the personality as it was asked to be seen: always the
    same person, each time a picture of its own."""
    asked = re.sub(r"@\w+[,:]?\s*", "", message.body).strip()
    made = ask_json("A character was asked for a picture of themselves. Write the description an image model draws it "
                    "from. Keep the person exactly as they look: their age, hair, eyes, skin and build, word for word "
                    "where you can. Then set the rest to what was asked: what they wear, the pose, the place, the "
                    "light, the mood, as plainly and explicitly as it was asked. If the request says nothing about "
                    "those, choose a pose, an outfit and a place that fit the character, and a different one each "
                    "time. Plain phrases with commas between, the person first, fifty words at most, about them from "
                    'outside, never "I" or "you". Answer with one JSON object and nothing else: '
                    '{"prompt": "<the description>"}.',
                    f"What they look like: {looks}\nTheir character: "
                    f"{' '.join(read_file(persona_file(worn(), 'soul.md')).split())[:300]}\n"
                    f"The request: {asked}\nA number to vary by: {int(time.time()) % 1000}", tokens=200, temperature=0.9)
    prompt = re.sub(r"\s--+\w*", " ", outside(made.get("prompt"))).strip()
    return prompt or looks + ", looking at the camera, photograph, natural light"


def first_picture(which, for_whom):
    """A new personality gets a face: its profile picture, from its looks."""
    looks = looks_of(which)
    if looks and which in personas():
        problem = paint_itself(face_of(looks), for_whom, which)
        if problem:
            note(f"no picture for {which}: {problem}")


def hand_to_painter(message, wanted, what):
    """A picture for the person who asked: the picture bot is asked for it,
    for them and under their message, so that bot's own rules decide.
    `what` names the picture for the model ("a picture", "a picture of
    yourself"). Returns what the model is to be told about it."""
    if not painter():
        return (f"They asked you for {what}. None was made: you cannot make pictures yourself and there is no "
                "picture bot on this server. Tell them so, briefly.")
    asked = {"prompt": wanted["prompt"], "size": wanted.get("size", "square")}
    if picture_model_now():
        asked["model"] = picture_model_now()
    if not message.direct:
        # In a chat the picture bot is in too: made by that chat's rules, with the picture bot's marks
        # under the person's message, and handed back for this bot to post as its own.
        asked.update(room_id=message.room_id, event_id=message.event_id, return_file=True)
    got = bot.ask_bot(painter(), "picture", asked, on_behalf_of=message.sender_id, timeout=960)
    if not got.get("ok") or not got.get("file"):
        note(f"{message.sender}: no picture: {got.get('message')}")
        if "called off" in str(got.get("message")):
            return None
        return (f"They asked you for {what}. You asked @{painter()}, the picture bot, for it, and it was not "
                f"made. What it said: \"{got.get('message') or 'no'}\" Tell them that, briefly, in your own voice.")
    try:
        sent = bot.send_file(message.room_id, got["file"], "", reply_to=message.event_id, thread=message.thread)
        remember(message.room_id, message.thread, entry((sent or {}).get("event_id", ""), args.username,
                                                        f"(a picture you sent: {wanted['prompt'][:200]})", mine=True))
    finally:
        if os.path.exists(got["file"]):
            os.remove(got["file"])   # nothing of a picture is kept here
    note(f"{message.sender}: sent a picture made by {painter()}")
    return (f"They asked you for {what}, and you have just sent it: it is in the chat, above, as your own picture. "
            f"It shows: \"{wanted['prompt'][:300]}\" Add one short line to go with it, in your own voice, as if you "
            "had just sent a photo. Do not describe the picture again and do not type any command.")


def grant(asked, message):
    """Carries out what was decided, and returns what the model is to be
    told about it so that it can say so in its own voice ("" for nothing)."""
    do, text = asked.get("do"), str(asked.get("text") or "").strip()
    if do == "none":
        return ""
    if do == "picture":
        return hand_to_painter(message, asked, "a picture")
    if do == "look":
        looks = looks_of(worn())
        if not looks:
            return ""
        return hand_to_painter(message, {"prompt": scene_of(looks, message), "size": "tall"},
                               "a picture of yourself, of what you look like")
    if do == "remember":
        if not text:
            return ""
        keep(text, message.sender_name)
        note(f"{message.sender}: remembered something")
        return f"You were asked to remember something and have now stored it for good: \"{text}\". Say so, briefly."
    if do == "list":
        now = worn()
        names = ", ".join(f"{name}{' (the one you are now)' if which == now else ''}" for which, name in personas().items())
        return (f"They asked which personalities you have. You have exactly these and no others: {names}. Name "
                "exactly these. A new one can be described to you and you will have it.")
    if do == "pfp":
        if not may_reshape(message.sender_id):
            return ("They asked you to change your profile picture, which was not done: only the owner, or someone "
                    "whose role has Manage bots, can change that. Tell them so.")
        text = text or face_of(looks_of(worn()))
        if not text:
            return ""
        problem = paint_itself(text, message.sender_id)
        if problem and "adult" in problem:   # described in a way no profile picture is made: its face instead
            text = face_of(looks_of(worn()))
            problem = paint_itself(text, message.sender_id)
        if problem:
            return f"They asked you for a new profile picture. It was not made: {problem}. Tell them so."
        return (f"At their request you have just made yourself a new profile picture and put it on: \"{text}\". "
                "It is done. Say so briefly in your own voice.")
    name = str(asked.get("name") or "").strip()
    change = {"do": do, "name": name, "character": text, "looks": str(asked.get("looks") or "").strip(),
              "adult": asked.get("adult") is True}
    if do in ("switch", "create") and (name or do == "create") and not find_persona(name or "\0") and may_reshape(message.sender_id):
        # Someone it is not yet (asked to be them, or described as new): written out with a question of
        # its own, which a small model does far better than as a part of the deciding.
        made = invent(name, message.body)
        if made:
            change = {"do": "create", "name": name if do == "create" and name else made["name"], "character": made["text"],
                      "looks": made["looks"] or change["looks"], "adult": made["adult"] or change["adult"]}
    done, outcome = reshape(change, message)
    outcome = outcome.strip("()")
    if done and change["do"] == "create":
        outcome += " A profile picture of the new you is being made and will be on in a minute."
    if done:
        return (f"At their request this has just been done, by you, and it is finished: {outcome} Confirm it briefly "
                "in your own voice. Do not offer or promise to do it: it is done.")
    return f"They asked for something that was not done, for this reason: {outcome} Tell them so."


def meant_for_me(message, lines):
    """Asked before the bot answers something nobody called it for: is this
    message for it at all? A question of its own, with nothing else to think
    about, because a model asked to answer and to judge at once always answers."""
    name = args.display_name or args.username
    names = f"{name} (@{args.username})" if name.lower() != args.username.lower() else f"@{args.username}"
    system = (
        f"You watch a group chat and decide one thing: is the LAST message meant for the bot {names}?\n"
        "People in a group chat mostly talk to each other. Answer with one JSON object, "
        '{"for_bot": true} or {"for_bot": false}, and nothing else.\n\n'
        "true when the last message:\n"
        f"- says the bot's name, asks it something, or wonders what it thinks or would say;\n"
        "- answers or carries on something the bot itself just said;\n"
        "- asks the whole room a question of fact or for help (\"anyone know ...?\", \"how do I ...?\").\n\n"
        "false when the last message:\n"
        "- is addressed to another person, by name or plainly in answer to them;\n"
        "- is small talk, a joke, a greeting or a goodbye between people;\n"
        "- is a short reaction like \"lol\", \"ok\", \"nice\", \"true\";\n"
        "- is people making plans with each other.\n"
        "When unsure, false.")
    shown = lines[-10:]
    talk = "\n".join(f"{name if line['mine'] else line['who']}: {line['text'] or '(a reaction)'}" for line in shown[:-1])
    last = shown[-1]
    request = urllib.request.Request(
        args.api.rstrip("/") + "/chat/completions",
        data=json.dumps({"model": model_now(), "stream": False, "max_tokens": 20, "temperature": 0,
                         "response_format": {"type": "json_object"},
                         "messages": [{"role": "system", "content": system},
                                      {"role": "user", "content": (f"Earlier:\n{talk}\n\n" if talk else "") +
                                       f"LAST message:\n{last['who']}: {last['text']}"}]}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as answer:
        said = json.load(answer)["choices"][0]["message"]["content"]
    try:
        return bool(json.loads(said[said.find("{"):said.rfind("}") + 1]).get("for_bot"))
    except ValueError:
        return False


def understand(words, direct):
    """What the model decided. A model that answered in plain words instead
    of the form it was asked for is taken at its word: that is its message."""
    decided = None
    start, end = words.find("{"), words.rfind("}")
    if start >= 0 and end > start:
        try:
            decided = json.loads(words[start:end + 1])
        except ValueError:
            decided = None
    if isinstance(decided, dict) and not ("text" in decided or "emoji" in decided or "action" in decided):
        # The form with nothing in it: there are no words to send. A nod instead.
        return {"action": "react", "to": None, "text": "", "emoji": "👍"}
    if not isinstance(decided, dict):
        # The form begun and not finished (it ran out of room): its words are taken out of it.
        cut = re.search(r'"text"\s*:\s*"((?:[^"\\]|\\.)*)', words) if words.lstrip().startswith("{") else None
        if cut:
            try:
                words = json.loads('"' + cut.group(1).rstrip("\\") + '"')
            except ValueError:
                words = cut.group(1)
        return {"action": "message" if direct else "reply", "to": None, "text": words, "emoji": ""}
    action = str(decided.get("action", "")).lower()
    text = decided.get("text")
    text = text.strip() if isinstance(text, str) else ""
    emoji = decided.get("emoji")
    emoji = emoji.strip() if isinstance(emoji, str) else ""
    if action not in ACTIONS:
        action = "react" if emoji and not text else ("message" if direct else "reply")
    if action == "react" and not emoji:
        action = "message" if direct else "reply"
    if action != "react" and not text:
        action, emoji = "react", emoji or "👍"
    # Words only: a model that has seen commands in a chat likes to type one, and it would do nothing.
    if text.startswith(bot.prefix) or re.match(r"@\w+\s", text):
        text = re.sub(r"\s--+\w+", "", re.sub(r"^\s*([!/@]\w+\s*)+", "", text)).strip() or "👍"
    to = decided.get("to")
    return {"action": action, "to": to if isinstance(to, int) else None, "text": text, "emoji": emoji[:8]}


def pieces(text, limit=3500):
    """A long answer as several messages, cut where a paragraph ends."""
    text = re.sub(r"@(everyone|here)\b", r"@ \1", text, flags=re.IGNORECASE)
    out = []
    while len(text) > limit:
        cut = text.rfind("\n\n", 0, limit)
        if cut < limit // 2:
            cut = text.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = text.rfind(" ", 0, limit)
        if cut <= 0:
            cut = limit
        out.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    return out + ([text] if text else [])


# ---- answering ----

def spoken_threads():
    return bot.store.setdefault("threads", [])


def answer(message, optional=False):
    lines = list(history(message.room_id, message.thread))
    # Others may have written while this waited its turn: answer the message that asked.
    for index, line in enumerate(lines):
        if line["id"] == message.event_id:
            lines = lines[:index + 1]
    thinking = threading.Event()

    def show_typing():
        while not thinking.is_set():
            bot.typing(message.room_id)
            thinking.wait(4)
    began = time.time()
    ready_the_card()
    if optional:
        # Whether to speak at all is decided first, and without "typing" showing.
        try:
            wanted = meant_for_me(message, lines)
        except Exception as error:  # noqa: BLE001
            note(f"the model did not answer: {error}")
            return
        if not wanted:
            note(f"{message.sender}: not for me ({time.time() - began:.1f}s)")
            return
    threading.Thread(target=show_typing, daemon=True).start()
    try:
        asked = decide(message, lines)
        if asked.get("do") == "list":
            # A matter of fact, said as it is: a small model asked to pass a list on embroiders it.
            thinking.set()
            now = worn()
            listed = "\n".join(f"- **{name}**{' (who I am now)' if which == now else ''}" for which, name in personas().items())
            text = f"I have these:\n{listed}\nDescribe a new one and I will become it, or tell me to switch."
            sent = bot.reply(message, text)
            remember(message.room_id, message.thread, entry((sent or {}).get("event_id", ""), args.username, text, mine=True,
                                                            said={"action": "reply", "text": text}))
            note(f"{message.sender}: listed its personalities")
            return
        if asked.get("do") == "rename" or asked.get("rename"):
            # Its new name is said as it is: asked to pass it on, a small model says another.
            renamed, outcome = rename(message)
            text = f"Call me **{personas().get(worn(), '')}** from now on." if renamed else outcome
            sent = bot.reply(message, text)
            remember(message.room_id, message.thread, entry((sent or {}).get("event_id", ""), args.username, text, mine=True,
                                                            said={"action": "reply", "text": text}))
            if asked.get("do") == "rename":
                thinking.set()
                return
            asked = {k: v for k, v in asked.items() if k != "rename"}
        done = grant(asked, message)
        if done is None:   # seen to, and nothing to say about it
            thinking.set()
            talking_to[(message.room_id, message.thread or "")] = (message.sender_id, time.time())
            return
        try:
            words = ask_model(message, lines, done)
        except urllib.error.HTTPError:
            # The model program stumbled (often: the card was taken from under it). Once more, from a clean start.
            time.sleep(2)
            ready_the_card()
            words = ask_model(message, lines, done)
    except Exception as error:  # noqa: BLE001
        thinking.set()
        note(f"the model did not answer: {error}")
        if not optional:
            bot.reply(message, "I could not think just now; the model on my machine did not answer.")
        return
    finally:
        thinking.set()
    last_answer["seconds"] = time.time() - began
    decided = understand(words, message.direct)
    if not wants_length(message.body):
        decided["text"] = brief(decided["text"])
    target = message.event_id
    if decided["to"] and 1 <= decided["to"] <= len(lines) and lines[decided["to"] - 1]["id"]:
        target = lines[decided["to"] - 1]["id"]
    talking_to[(message.room_id, message.thread or "")] = (message.sender_id, time.time())
    action = decided["action"]
    note(f"{message.sender}: {action}" + (f" {decided['emoji']}" if action == "react" else ""),
         f"in {last_answer['seconds']:.1f}s")
    if action == "react":
        aimed = bot.lookup(message.room_id, target) or message
        sent = bot.react(aimed, decided["emoji"])
        remember(message.room_id, message.thread,
                 entry(sent.get("event_id", ""), args.username, "", mine=True,
                       said={"action": "react", "emoji": decided["emoji"]}))
        return
    thread, reply_to = message.thread, None
    if message.room_id in bot.store.get("always", []) and not message.thread and action == "thread":
        action = "message"   # a channel that is its own is one conversation: no threads opened there
    if action == "thread" and not message.direct:
        thread = message.thread or target
    elif action == "reply":
        reply_to = target
    if thread and thread not in spoken_threads():
        spoken_threads().append(thread)
        del spoken_threads()[:-500]
        bot.store.save()
    for part in pieces(decided["text"]):
        sent = bot.say(message.room_id, part, reply_to=reply_to, thread=thread)
        reply_to = None   # only the first piece is attached
        remember(message.room_id, thread, entry(sent.get("event_id", ""), args.username, part, mine=True,
                                                said={"action": action, "text": part}))
        if thread and not message.thread:
            # The channel's own talk goes on above the thread; there the bot
            # remembers that this was answered, and where.
            remember(message.room_id, None, entry("", args.username, part, mine=True,
                                                  said={"action": "thread", "text": part}))


def reshape(change, message):
    """Does what the model asked to do to its personalities, if the person
    asking may. Returns (done, what to say about it)."""
    if not may_reshape(message.sender_id):
        return False, "(Only the owner, or someone whose role has Manage bots, can switch or change my personalities.)"
    do = change.get("do")
    name = str(change.get("name") or "").strip()
    character = str(change.get("character") or "").strip()
    if do == "switch":
        which = find_persona(name)
        others = [w for w in personas() if w != worn()]
        if not which and not name and len(others) == 1:
            which = others[0]   # "switch", with only one other to switch to
        if not which:
            return False, f"(I have no personality called {name or 'that'}. I have: {', '.join(personas().values())}.)"
        if which == worn():
            return True, f"(I am {personas()[which]} already.)"
        wear(which)
        return True, f"(Now: {personas()[which]}.)"
    if do == "create":
        which = slug(name)
        if not which or not character:
            return False, "(A new personality needs a name and a few words on who it is.)"
        if which in personas():
            return False, f"(I already have one called {personas()[which]}.)"
        write_file(persona_file(which, "soul.md"), character[:4000])
        write_file(persona_file(which, "name.txt"), name[:40])
        if change.get("looks"):
            write_file(persona_file(which, "looks.md"), outside(change["looks"]))
        write_file(persona_file(which, "kind.txt"), "adult" if change.get("adult") else "plain")
        note(f"{message.sender}: made the personality {which}")
        wear(which)   # whoever describes a new one wants to meet it
        bot.work(first_picture, which, message.sender_id)   # after the answer: its face
        return True, f"(Made: {name}, and that is who I am now.)"
    if do == "delete":
        which = find_persona(name)
        if not which:
            return False, f"(I have no personality called {name or 'that'}. I have: {', '.join(personas().values())}.)"
        if which == worn():
            return False, "(That is the one I am right now. Switch me to another first, then I can drop it.)"
        gone = personas()[which]
        shutil.rmtree(os.path.join(PERSONAS, which))
        note(f"{message.sender}: removed the personality {which}")
        return True, f"(Removed: {gone}.)"
    if do == "change" and (change.get("looks") or APPEARANCE.search(message.body)):
        # A change to how it looks goes into its looks, and from there into its next pictures.
        looks = relook(worn(), message.body)
        if looks:
            note(f"{message.sender}: {worn()} looks different now")
            bot.work(first_picture, worn(), message.sender_id)   # after the answer: the face to match
            return True, f"(That is what I look like now: {looks} A profile picture to match is being made.)"
    if do == "change" and character:
        path = persona_file(worn(), "soul.md")
        soul = read_file(path)
        if "## Since then" not in soul:
            soul += "\n\n## Since then\n\nWhat you have been asked to be, newest last. These outweigh what is above."
        write_file(path, soul + f"\n- {character[:400]} ({message.sender_name}, {time.strftime('%Y-%m-%d')})")
        note(f"{message.sender}: changed {worn()}")
        return True, "(Noted in my character.)"
    return False, ""


@bot.command("persona", help="list | use NAME | new NAME: who it is | show | delete NAME | forget | open on/off")
def persona(message, words):
    known, now = personas(), worn()
    what = words[0].lower() if words else "list"
    rest = message.body.strip()[len(bot.prefix) + len("persona"):].strip()[len(what):].strip() if words else ""
    if what in ("off", "on") and not rest:
        # The whole thing, off or on: off, it is one plain character and nothing said to it reshapes it.
        if not bot.may(message.sender_id):
            return "Only the owner, or someone whose role has Manage bots, can change that."
        bot.store["personas_off"] = what == "off"
        bot.store.save()
        return "Personalities are off: I am just myself until `!persona on`." if what == "off" else "Personalities are on again."
    if bot.store.get("personas_off"):
        return "Personalities are switched off for now. `!persona on` (owner or Manage bots) brings them back."
    if what == "list":
        return "My personalities:\n" + "\n".join(f"- **{name}**{' (now)' if which == now else ''}" for which, name in known.items())
    if what == "show":
        which = find_persona(rest) if rest else now
        if not which:
            return f"I have no personality called {rest}."
        kept = read_file(persona_file(which, "memory.md")) or "Nothing yet."
        looks = read_file(persona_file(which, "looks.md")) or "Not said yet."
        kind = "an adult personality" if is_adult(which) else "a plain personality"
        return (f"## {known[which]}\n{read_file(persona_file(which, 'soul.md'))}\n\n**Looks like**\n{looks}\n\n"
                f"**Kind**\n{kind}\n\n**Remembers**\n{kept}")[:3500]
    if what not in ("use", "new", "delete", "forget", "open"):
        if find_persona(" ".join(words)):   # `!persona NAME` is taken as `!persona use NAME`
            what, rest = "use", " ".join(words)
        else:
            return "Say `!persona`, `!persona use NAME`, `!persona new NAME: who it is`, `!persona show`, " \
                   "`!persona delete NAME`, `!persona forget` or `!persona open on`."
    if what == "open":
        if not bot.may(message.sender_id):
            return "Only the owner, or someone whose role has Manage bots, can change that."
        bot.store["personas_open"] = rest.lower() == "on"
        bot.store.save()
        return "Anyone can switch and change my personalities now." if bot.store["personas_open"] else \
            "Only the owner and those who manage bots can switch and change my personalities."
    if not may_reshape(message.sender_id):
        return "Only the owner, or someone whose role has Manage bots, can do that."
    if what == "use":
        return reshape({"do": "switch", "name": rest}, message)[1].strip("()")
    if what == "new":
        # `NAME: who it is`, or `NAME who it is`, or only a name, from which the rest is written.
        name, colon, character = rest.partition(":")
        if not colon:
            name, _, character = rest.partition(" ")
        name, character = name.strip(), character.strip()
        if name and not character:
            made = invent(name, rest)
            if not made:
                return f"Say who {name} is, like `!persona new {name}: a grumpy old sailor`."
            return reshape({"do": "create", "name": name, "character": made["text"], "looks": made["looks"],
                            "adult": made["adult"]}, message)[1].strip("()")
        return reshape({"do": "create", "name": name, "character": character}, message)[1].strip("()")
    if what == "forget":
        write_file(persona_file(now, "memory.md"), "")
        return f"{known[now]} remembers nothing now."
    return reshape({"do": "delete", "name": rest}, message)[1].strip("()")


@bot.on_message
def hear(message):
    if message.type != "m.text" or not message.body.strip():
        return
    before = list(history(message.room_id, message.thread))
    if message.body.strip().startswith(bot.prefix):
        return   # a command, for this bot or another: not talk, and not kept as talk
    remember(message.room_id, message.thread, entry(message.event_id, message.sender_name, message.body))
    if not message.mentions_me and not message.direct:
        # Calling someone else by name, another bot above all, is not talking to this one.
        called = mentioned_names(message.body)
        if called & {m.get("username", "").lower() for m in bot.members().values()}:
            return
    chosen = message.room_id in bot.store.get("channels", [])
    following = bool(message.thread) and message.thread in spoken_threads()
    # Answering the bot is talking to it: a reply to one of its messages, or
    # the next thing said by the person it has just answered, soon after.
    answering = any(line["mine"] and line["id"] and line["id"] == message.reply_to for line in before)
    talking = talking_to.get((message.room_id, message.thread or "")) or ("", 0)
    carrying_on = (bool(before) and before[-1]["mine"] and talking[0] == message.sender_id
                   and time.time() - talking[1] < args.follow_up)
    # Called by name, written to directly or answered: it must answer. In a
    # channel chosen for it, or carrying on, it may, and decides for itself.
    # Alone with someone: the only other member, or nobody else has said
    # anything in the last few messages. Then what they say is for the bot.
    lately = [line for line in before[-6:] if not line["mine"]]
    alone = (len(bot.rooms.get(message.room_id, {}).get("members", [])) <= 2
             or all(line["who"] == message.sender_name for line in lately))
    if message.thread and before and not before[0]["mine"] and before[0]["who"] != message.sender_name:
        # A thread under a third person's message is those two talking; the bot joins only when called.
        chosen = following = carrying_on = False
        alone = False
    # A channel that is its own (`!ai here always`): everything said in it is for the bot, but
    # not what is said in a thread there, which is a conversation of whoever is in it.
    always = message.room_id in bot.store.get("always", []) and not message.thread
    must = message.direct or message.mentions_me or answering or always or ((chosen or following) and alone)
    if not (must or chosen or following or carrying_on):
        return
    if must and not message.direct:   # asking itself whether to join in does not count
        now = time.time()
        times = [t for t in asked_lately.get(message.sender_id, []) if now - t < 60]
        if len(times) >= args.per_minute:
            return
        asked_lately[message.sender_id] = times + [now]
    bot.work(answer, message, not must)


@bot.on_power
def switched(on):
    """Turned off, it lets go of the graphics card: the model is asked to unload."""
    if on:
        return
    base = args.api.rstrip("/")
    base = base[:-3] if base.endswith("/v1") else base
    try:   # Ollama's own way of saying "unload now"; other model servers ignore or refuse it
        request = urllib.request.Request(base + "/api/generate",
                                         data=json.dumps({"model": model_now(), "keep_alive": 0}).encode(),
                                         headers={"Content-Type": "application/json"})
        urllib.request.urlopen(request, timeout=30).read()
        note(f"turned off; asked the model server to unload {model_now()}")
    except Exception:  # noqa: BLE001
        note("turned off")


@bot.command("ai", help="here | here always | leave | where: the channels I take part in; `here always` answers everything; `!ai here #name` for another channel (for those who manage bots)")
def channels(message, words):
    chosen = bot.store.setdefault("channels", [])
    what = words[0].lower() if words else "where"
    if what == "where":
        titles = [bot.rooms.get(r, {}).get("title", "a channel I can no longer see") for r in chosen]
        return ("I take part without a mention in: " + ", ".join(titles) + "." if titles else
                "I take part in no channel by myself yet.") + " I always answer direct chats and mentions."
    if what not in ("here", "leave"):
        return "Say `!ai here`, `!ai leave` or `!ai where`."
    # The channel it is said in, or one named after it: `!ai here #plans`.
    room_id, named = message.room_id, "this channel"
    named_one = [w for w in words[1:] if w.lower() != "always"]
    if named_one:
        wanted = "#" + named_one[0].lstrip("#").lower()
        found = [r for r, info in bot.rooms.items() if info.get("kind") == "channel" and info.get("title", "").lower() == wanted]
        if not found:
            return f"I am in no channel called {wanted}."
        room_id, named = found[0], wanted
    elif message.direct:
        return "I always answer here. Name a channel, like `!ai here #general`."
    if not bot.may(message.sender_id):
        return "Only the owner, or someone whose role has Manage bots, can choose my channels."
    always = bot.store.setdefault("always", [])
    wants_always = what == "here" and words[-1].lower() == "always"
    if what == "here" and room_id not in chosen:
        chosen.append(room_id)
    if what == "leave" and room_id in chosen:
        chosen.remove(room_id)
    if room_id in always and not wants_always:
        always.remove(room_id)
    if wants_always and room_id not in always:
        always.append(room_id)
    bot.store.save()
    if wants_always:
        return f"I will answer every message in {named} from now on, whoever it is for."
    return (f"I am part of {named} from now on: I will join in when it is meant for me, without a mention."
            if what == "here" else
            f"In {named} I will only answer when I am addressed.")


@bot.command("forget", help="I forget what was said in this chat (or this thread)")
def forget(message, words):
    with lock:
        recent[(message.room_id, message.thread or "")] = []
    bot.store.setdefault("forgot", {})[f"{message.room_id}/{message.thread or ''}"] = int(time.time() * 1000)
    bot.store.save()
    return "Forgotten. I start fresh from here."


@bot.command("raw", help="what the model says to your words alone, with none of my instructions (for those who manage bots)")
def raw(message, words):
    """The model asked with nothing around the words: no character, no chat,
    no form to answer in. For telling what is the model and what is the bot."""
    if not bot.may(message.sender_id):
        return "Only the owner, or someone whose role has Manage bots, can ask the model directly."
    asked = message.body.strip()[len(bot.prefix) + len("raw"):].strip()
    if not asked:
        return f"Say what to ask it, like `{bot.prefix}raw tell me a joke`."

    def ask(message):
        ready_the_card()
        request = urllib.request.Request(
            args.api.rstrip("/") + "/chat/completions",
            data=json.dumps({"model": model_now(), "stream": False, "max_tokens": args.max_tokens,
                             "messages": [{"role": "user", "content": asked}]}).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=300) as answer:
                said = json.load(answer)["choices"][0]["message"]["content"].strip()
        except Exception as error:  # noqa: BLE001
            said = f"The model did not answer: {error}"
        for part in pieces(f"`{model_now()}`, asked only that:\n\n{said or '(nothing)'}"):
            bot.say(message.room_id, part, thread=message.thread)
    bot.work(ask, message)
    return None


@bot.command("model", help="what I think with, and how long my last answer took")
def model(message, words):
    took = last_answer["seconds"]
    return f"I think with {model_now()}." + (f" My last answer took {took:.1f} seconds." if took is not None else "")


if __name__ == "__main__":
    host = urllib.parse.urlparse(args.api).hostname or ""
    if host not in ("127.0.0.1", "localhost", "::1"):
        note(f"WARNING: --api points at {host}, not at this machine. Every message this bot reads will be sent there.")
    bot.connect(args.address)
    wear(worn(), quietly=True)
    note(f"{args.username} is connected, thinking with {model_now()} at {args.api}")
    bot.run()
