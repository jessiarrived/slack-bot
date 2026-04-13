from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv
from openai import OpenAI
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from supabase import create_client, Client
from flask import Flask
import os
import re
import threading


def slack_api_error_payload(e: SlackApiError) -> dict:
    """Normalize Slack SDK error response to a dict (error, needed, etc.)."""
    r = getattr(e, "response", None)
    if isinstance(r, dict):
        return r
    data = getattr(r, "data", None) if r is not None else None
    return data if isinstance(data, dict) else {}


def create_app() -> App:
    load_dotenv()

    bot_token = (os.getenv("SLACK_BOT_TOKEN") or "").strip()
    user_token = (os.getenv("SLACK_USER_TOKEN") or "").strip()
    app_token = (os.getenv("SLACK_APP_TOKEN") or "").strip()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")

    if not bot_token or not app_token:
        raise RuntimeError(
            "Missing SLACK_BOT_TOKEN or SLACK_APP_TOKEN. "
            "Set them in your .env file."
        )
    if not user_token:
        raise RuntimeError(
            "Missing SLACK_USER_TOKEN (User OAuth token, xoxp-). "
            "Required for conversations.history and conversations.replies. "
            "Set it in your .env file."
        )

    if not openai_api_key:
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Set it in your .env file."
        )

    app = App(token=bot_token)
    # User token: history/replies (channel/private metadata scopes on the user token).
    user_slack = WebClient(token=user_token, timeout=15)

    if not str(bot_token).startswith("xoxb-"):
        print(
            "[slack] warning: SLACK_BOT_TOKEN should start with xoxb- (bot token). "
            "If scopes fail, fix Railway / .env."
        )
    if not str(user_token).startswith("xoxp-"):
        print(
            "[slack] warning: SLACK_USER_TOKEN should start with xoxp- (user OAuth token). "
            "History API calls will fail with a user-install token in the wrong variable."
        )

    try:
        ut = user_slack.auth_test()
        print(
            "[slack-user] auth_test ok — history is read as this user: "
            f"user_id={ut.get('user_id')!r} team={ut.get('team')!r}"
        )
    except SlackApiError as e:
        _d = slack_api_error_payload(e)
        print(
            "[slack-user] auth_test FAILED — user token cannot call Slack API: "
            f"error={_d.get('error')!r} needed={_d.get('needed')!r}. "
            "Fix SLACK_USER_TOKEN and User Scopes, then reinstall the app to the workspace."
        )
    except Exception as e:
        print(f"[slack-user] auth_test unexpected error: {type(e).__name__}: {e}")
    # Keep timeouts short so the bot can't hang silently
    client = OpenAI(api_key=openai_api_key, timeout=20.0, max_retries=1)

    # Identify this bot's user ID so we can avoid double-handling messages
    auth_info = app.client.auth_test()
    bot_user_id = auth_info.get("user_id")

    # Table name must exist in your project (default was `messages`; many use `slack_messages`).
    supabase_messages_table = (os.getenv("SUPABASE_MESSAGES_TABLE") or "messages").strip()

    supabase: Client | None = None
    if supabase_url and supabase_key:
        supabase = create_client(supabase_url, supabase_key)
        print(
            f"[supabase] client configured; inserts go to table={supabase_messages_table!r}"
        )
    else:
        _missing = [n for n, v in (
            ("SUPABASE_URL", supabase_url),
            ("SUPABASE_SERVICE_ROLE_KEY or SUPABASE_ANON_KEY", supabase_key),
        ) if not v]
        print(
            "[supabase] message logging is OFF until Railway Variables include: "
            + ", ".join(_missing)
        )

    def log_message(source: str, user: str, channel: str, text: str, ts):
        """
        Persist a Slack message to Supabase if configured.
        Table name: SUPABASE_MESSAGES_TABLE (default: messages).
        Expects columns: source, user_id, channel_id, text, ts (all text-compatible).
        """
        if not supabase or not text:
            return
        try:
            supabase.table(supabase_messages_table).insert(
                {
                    "source": source,
                    "user_id": user,
                    "channel_id": channel,
                    "text": text,
                    "ts": ts or "",
                }
            ).execute()
        except Exception as e:
            extra = ""
            for name in ("details", "hint", "code", "message"):
                if hasattr(e, name):
                    val = getattr(e, name)
                    if val is not None and str(val) and str(val) != str(e):
                        extra += f" {name}={val!r}"
            print(
                f"[supabase] insert failed (table={supabase_messages_table!r}): "
                f"{type(e).__name__}: {e}{extra}"
            )

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

    def _extract_text_from_blocks(blocks) -> str:
        if not isinstance(blocks, list):
            return ""
        parts: list[str] = []

        def walk(obj):
            if isinstance(obj, dict):
                if obj.get("type") in ("plain_text", "mrkdwn"):
                    t = obj.get("text")
                    if isinstance(t, str) and t.strip():
                        parts.append(t.strip())
                        return
                # rich_text_section / broadcast / emoji etc. use type "text" for runs
                if obj.get("type") == "text":
                    t = obj.get("text")
                    if isinstance(t, str) and t.strip():
                        parts.append(t.strip())
                        return
                if obj.get("type") == "link":
                    t = obj.get("text") or obj.get("url")
                    if isinstance(t, str) and t.strip():
                        parts.append(t.strip())
                        return
                text_obj = obj.get("text")
                if isinstance(text_obj, dict):
                    walk(text_obj)
                elif isinstance(text_obj, str) and text_obj.strip():
                    parts.append(text_obj.strip())
                for key in ("elements", "fields"):
                    nested = obj.get(key)
                    if isinstance(nested, list):
                        for item in nested:
                            walk(item)
            elif isinstance(obj, list):
                for item in obj:
                    walk(item)

        for block in blocks:
            walk(block)
        return " ".join(parts).strip()

    def _slack_message_plain_text(message: dict) -> str:
        text = (message.get("text") or "").strip()
        if text:
            return text
        blocks = message.get("blocks")
        if blocks:
            extracted = _extract_text_from_blocks(blocks)
            if extracted:
                return extracted
        for att in message.get("attachments") or []:
            if not isinstance(att, dict):
                continue
            ft = (att.get("fallback") or att.get("text") or "").strip()
            if ft:
                return ft
        return ""

    USER_TOKEN_LABEL = "user_oauth(xoxp)"
    BOT_TOKEN_LABEL = "bot(xoxb)"

    def log_slack_api_error(where: str, token_kind: str, method: str, e: SlackApiError) -> None:
        data = slack_api_error_payload(e)
        err = data.get("error")
        needed = data.get("needed")
        extra = ""
        for key in ("warning", "response_metadata"):
            if data.get(key):
                extra += f" {key}={data[key]!r}"
        print(
            f"[slack-api] where={where} method={method} token={token_kind} "
            f"error={err!r} needed={needed!r}{extra}"
        )

    def log_history_context_hint(payload: dict) -> None:
        err = payload.get("error")
        if err == "not_in_channel":
            print(
                "[history] hint: not_in_channel — this token's user/bot must be a member of the "
                "channel. For SLACK_USER_TOKEN, the person who installed the app must be invited "
                "to the channel (inviting only the bot is not enough for the user token)."
            )
        elif err == "missing_scope":
            print(
                "[history] hint: missing_scope — add the scopes Slack returns in `needed` under "
                "OAuth & Permissions → User Token Scopes (for xoxp-) or Bot Token Scopes, "
                "then reinstall the app to the workspace."
            )
        elif err in ("invalid_auth", "token_revoked", "account_inactive"):
            print(
                f"[history] hint: {err} — rotate the token in Slack app settings and update "
                "Railway Variables / .env."
            )

    def fetch_conversations_replies_paginated(
        slack_client: WebClient,
        channel: str,
        thread_ts: str,
        *,
        max_messages: int = 200,
        page_size: int = 100,
    ) -> list[dict]:
        """Cursor-paginated conversations.replies via user token (Slack max 100 per call)."""
        messages: list[dict] = []
        cursor: str | None = None
        while len(messages) < max_messages:
            page_limit = min(page_size, max_messages - len(messages))
            kwargs: dict = {"channel": channel, "ts": thread_ts, "limit": page_limit}
            if cursor:
                kwargs["cursor"] = cursor
            resp = slack_client.conversations_replies(**kwargs)
            batch = list(resp.get("messages") or [])
            messages.extend(batch)
            cursor = (resp.get("response_metadata") or {}).get("next_cursor") or None
            if not cursor or not batch:
                break
        return messages

    def fetch_conversations_history_paginated(
        slack_client: WebClient,
        channel: str,
        latest_ts: str,
        *,
        max_messages: int = 200,
        page_size: int = 100,
    ) -> list[dict]:
        """Cursor-paginated conversations.history via user token (Slack max 100 per call)."""
        messages_newest_first: list[dict] = []
        cursor: str | None = None
        while len(messages_newest_first) < max_messages:
            page_limit = min(page_size, max_messages - len(messages_newest_first))
            kwargs: dict = {"channel": channel, "limit": page_limit}
            if cursor:
                kwargs["cursor"] = cursor
            else:
                kwargs["latest"] = latest_ts
                kwargs["inclusive"] = True
            resp = slack_client.conversations_history(**kwargs)
            batch = list(resp.get("messages") or [])
            messages_newest_first.extend(batch)
            cursor = (resp.get("response_metadata") or {}).get("next_cursor") or None
            if not cursor or not batch:
                break
        # Present oldest first to the model.
        return list(reversed(messages_newest_first))

    def build_history_prompt(channel: str, ts: str, thread_ts, fallback_text: str) -> str:
        """Fetch recent channel/thread history and return a prompt string for the LLM."""
        method = "conversations.replies" if thread_ts else "conversations.history"
        messages: list[dict] = []
        try:
            try:
                if thread_ts:
                    messages = fetch_conversations_replies_paginated(
                        user_slack, channel, thread_ts
                    )
                else:
                    messages = fetch_conversations_history_paginated(user_slack, channel, ts)
            except SlackApiError as e_user:
                log_slack_api_error("history_context", USER_TOKEN_LABEL, method, e_user)
                log_history_context_hint(slack_api_error_payload(e_user))
                print("[history] retrying fetch with bot token…")
                try:
                    if thread_ts:
                        messages = fetch_conversations_replies_paginated(
                            app.client, channel, thread_ts
                        )
                    else:
                        messages = fetch_conversations_history_paginated(
                            app.client, channel, ts
                        )
                    print("[history] bot token fetch succeeded (user token failed).")
                except SlackApiError as e_bot:
                    log_slack_api_error("history_context", BOT_TOKEN_LABEL, method, e_bot)
                    log_history_context_hint(slack_api_error_payload(e_bot))
                    return fallback_text

            lines = []
            for m in messages:
                m_user = m.get("user") or m.get("bot_id") or "system"
                m_text = _slack_message_plain_text(m)
                if not m_text:
                    continue
                lines.append(f"{m_user}: {m_text}")

            if lines:
                print(f"[history] using {len(lines)} slack message(s) for context")
                return (
                    "Here is the recent Slack conversation history (oldest to newest):\n"
                    + "\n".join(lines)
                    + "\n\nRespond as the assistant to the most recent user message."
                )

            raw_count = len(messages)
            if raw_count:
                print(
                    f"[history] slack returned {raw_count} message(s) but none had extractable text "
                    "(try plain text or check Block Kit / attachments)."
                )
            else:
                print(
                    "[history] no messages returned (empty channel window or ts out of range). "
                    f"channel={channel!r} thread_ts={thread_ts!r}"
                )
        except Exception as e:
            print(f"[history] failed to fetch conversation history: {type(e).__name__}: {e}")
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
        except SlackApiError as e:
            log_slack_api_error("chat_reply", BOT_TOKEN_LABEL, "chat.postMessage", e)
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

        # Only auto-respond in DMs (1:1 or group), not in public/private channels.
        # Prefer Slack's channel_type when present; fall back to legacy D-prefix IM IDs.
        channel_type = message.get("channel_type")
        if channel_type:
            if channel_type not in ("im", "mpim"):
                return
        elif not channel or not channel.startswith("D"):
            return

        # Ignore messages without text, or from bots (including this bot)
        if not text:
            return
        if message.get("bot_id") or message.get("subtype") == "bot_message":
            return

        if not user:
            return

        history_prompt = build_history_prompt(channel, ts, thread_ts, text)

        reply = generate_ai_reply(history_prompt)
        try:
            say(f"<@{user}> {reply}")
            print("[slack] reply sent")
        except SlackApiError as e:
            log_slack_api_error("chat_reply", BOT_TOKEN_LABEL, "chat.postMessage", e)
        except Exception as e:
            print(f"[slack] failed to send reply: {type(e).__name__}: {e}")

    return app


if __name__ == "__main__":
    print("[slack-bot] Starting… (Socket Mode)")
    print(
        "[slack-bot] Responds to: @mentions in channels, and DMs (1:1 + group). "
        "Invite the bot to the channel for mentions."
    )
    slack_app = create_app()
    handler = SocketModeHandler(slack_app, os.environ["SLACK_APP_TOKEN"])

    port = os.environ.get("PORT")
    if port:
        # Railway (and similar) set PORT and expect a listening HTTP process for health checks.
        def _run_socket_mode() -> None:
            print("[slack-bot] Socket Mode worker started.")
            handler.start()

        threading.Thread(target=_run_socket_mode, daemon=True).start()

        health_app = Flask(__name__)

        @health_app.get("/")
        def _health():
            return "ok", 200

        print(
            f"[slack-bot] HTTP health on 0.0.0.0:{port} (Railway). "
            "Socket Mode runs in background."
        )
        health_app.run(host="0.0.0.0", port=int(port), threaded=True)
    else:
        print("[slack-bot] Connected. Waiting for events (Ctrl+C to stop).")
        handler.start()