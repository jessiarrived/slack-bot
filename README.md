## slack-bot

Slack bot using Slack Bolt (Socket Mode) with an LLM backend.

### Setup

- Create a virtualenv and install dependencies:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

- Create your env file:

```bash
cp .env.example .env
```

- Run the bot:

```bash
.venv/bin/python app.py
```

### Claude API smoke test

```bash
.venv/bin/python claude_smoke_test.py
```

