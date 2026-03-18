from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client, Client
import os
import re


def create_app() -> App:
    load_dotenv()

    bot_token = os.getenv("SLACK_BOT_TOKEN")
    app_token = os.getenv("SLACK_APP_TOKEN")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")

    if not bot_token or not app_token:
        raise RuntimeError(
            "Missing SLACK_BOT_TOKEN or SLACK_APP_TOKEN. "
            "Set them in your .env file."
        )

    if not openai_api_key:
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Set it in your .env file."
        )

    app = App(token=bot_token)
    # Keep timeouts short so the bot can't hang silently
    client = OpenAI(api_key=openai_api_key, timeout=20.0, max_retries=1)

    # Identify this bot's user ID so we can avoid double-handling messages
    auth_info = app.client.auth_test()
    bot_user_id = auth_info.get("user_id")

    supabase: Client | None = None
    if supabase_url and supabase_key:
        supabase = create_client(supabase_url, supabase_key)

    def log_message(source: str, user: str, channel: str, text: str, ts):
        """
        Persist a Slack message to Supabase if configured.
        Expects a table `messages` with at least:
          source (text), user_id (text), channel_id (text), text (text), ts (text)
        """
        if not supabase or not text:
            return
        try:
            supabase.table("messages").insert(
                {
                    "source": source,
                    "user_id": user,
                    "channel_id": channel,
                    "text": text,
                    "ts": ts or "",
                }
            ).execute()
        except Exception as e:
            print(f"[supabase] failed to log message: {type(e).__name__}: {e}")

    def generate_ai_reply(prompt: str) -> str:
        try:
            print("[llm] requesting completion...")
            response = client.chat.completions.create(
                model=openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant responding inside Slack. "
                            "You are given recent conversation history in the prompt; "
                            "do not say that you cannot see message history. "
                            "If needed, explain that you can only use the messages "
                            "that are shown to you in the prompt. "
                            "Do not start your reply by addressing the user by name or with an @mention—the message will be directed to them automatically. "
                            "Keep replies concise and friendly."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            reply = response.choices[0].message.content.strip()
            # Avoid double mention: remove leading <@USERID> if the LLM added one
            reply = re.sub(r"^\s*<@[A-Z0-9]+>\s*[,:]?\s*", "", reply)
            print("[llm] completion received")
            return reply
        except Exception as e:
            # Fallback so the bot still replies when the LLM call fails
            print(f"[llm] error: {type(e).__name__}: {e}")
            return (
                "I tried to call the AI model but hit an error "
                f"(`{type(e).__name__}`: {e}). "
                "Please check your OpenAI plan, billing, and API key settings."
            )

    def build_history_prompt(channel: str, ts: str, thread_ts, fallback_text: str) -> str:
        """Fetch recent channel/thread history and return a prompt string for the LLM."""
        try:
            if thread_ts:
                history = app.client.conversations_replies(
                    channel=channel,
                    ts=thread_ts,
                    limit=30,
                )
            else:
                history = app.client.conversations_history(
                    channel=channel,
                    latest=ts,
                    limit=30,
                    inclusive=True,
                )
            messages = history.get("messages", [])
            lines = []
            for m in messages:
                m_user = m.get("user") or m.get("bot_id") or "system"
                m_text = m.get("text", "")
                if not m_text:
                    continue
                lines.append(f"{m_user}: {m_text}")
            if lines:
                return (
                    "Here is the recent Slack conversation history (oldest to newest):\n"
                    + "\n".join(lines)
                    + "\n\nRespond as the assistant to the most recent user message."
                )
        except Exception as e:
            print(f"Failed to fetch conversation history: {e}")
        return fallback_text

    @app.event("app_mention")
    def handle_app_mention(body, say):
        event = body.get("event", {})
        text = event.get("text", "")
        user = event.get("user", "")
        channel = event.get("channel")
        ts = event.get("ts")
        thread_ts = event.get("thread_ts")

        print(f"[app_mention] channel={channel} user={user} text={text!r}")

        log_message("app_mention", user, channel, text, ts)

        cleaned = re.sub(r"<@[^>]+>", "", text).strip()
        if not cleaned:
            cleaned = "Say hello and explain what you can do."

        history_prompt = build_history_prompt(channel, ts, thread_ts, cleaned)
        reply = generate_ai_reply(history_prompt)
        try:
            say(f"<@{user}> {reply}")
            print("[slack] reply sent")
        except Exception as e:
            print(f"[slack] failed to send reply: {type(e).__name__}: {e}")

    @app.message(re.compile(".*"))
    def handle_any_message(message, say):
        user = message.get("user", "")
        text = message.get("text", "")
        channel = message.get("channel")
        thread_ts = message.get("thread_ts")
        ts = message.get("ts")

        print(
            f"[message] channel={channel} user={user} ts={ts} "
            f"thread_ts={thread_ts} text={text!r}"
        )

        log_message("dm", user, channel, text, ts)

        # Only auto-respond in direct messages (IMs).
        # Channel IDs starting with "D" are 1:1 DMs.
        if not channel or not channel.startswith("D"):
            return

        # Ignore messages without text, or from bots (including this bot)
        if not text:
            return
        if message.get("bot_id") or message.get("subtype") == "bot_message":
            return

        if not user:
            return

        # Try to fetch recent history (thread-first, then channel) so the AI can see context
        history_prompt = text
        try:
            if thread_ts:
                # We're in a thread – fetch that thread's messages
                history = app.client.conversations_replies(
                    channel=channel,
                    ts=thread_ts,
                    limit=30,
                )
            else:
                # Channel-level conversation
                history = app.client.conversations_history(
                    channel=channel,
                    latest=ts,
                    limit=30,
                    inclusive=True,
                )

            messages = history.get("messages", [])
            lines = []
            for m in messages:
                m_user = m.get("user") or m.get("bot_id") or "system"
                m_text = m.get("text", "")
                if not m_text:
                    continue
                lines.append(f"{m_user}: {m_text}")

            if lines:
                history_prompt = (
                    "Here is the recent Slack conversation history (oldest to newest):\n"
                    + "\n".join(lines)
                    + "\n\nRespond as the assistant to the most recent user message."
                )
        except Exception as e:
            # If we can't fetch history (missing scopes, etc.), just fall back to the single message
            print(f"Failed to fetch conversation history: {e}")
            history_prompt = text

        reply = generate_ai_reply(history_prompt)
        try:
            say(f"<@{user}> {reply}")
            print("[slack] reply sent")
        except Exception as e:
            print(f"[slack] failed to send reply: {type(e).__name__}: {e}")

    return app


if __name__ == "__main__":
    slack_app = create_app()
    handler = SocketModeHandler(slack_app, os.environ["SLACK_APP_TOKEN"])
    handler.start()