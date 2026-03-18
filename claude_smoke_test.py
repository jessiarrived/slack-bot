import os

from dotenv import load_dotenv


def main() -> None:
    load_dotenv()

    api_key = os.getenv("ANTHROPIC_API_KEY")
    model = os.getenv("ANTHROPIC_MODEL") or "claude-3-5-sonnet-latest"

    if not api_key:
        raise SystemExit(
            "Missing ANTHROPIC_API_KEY in .env (add it, then re-run)."
        )

    try:
        from anthropic import Anthropic
    except Exception as e:
        raise SystemExit(
            "Anthropic SDK not installed. Run: pip install -r requirements.txt\n"
            f"Import error: {type(e).__name__}: {e}"
        )

    client = Anthropic(api_key=api_key)

    # Keep output minimal and never print secrets.
    resp = client.messages.create(
        model=model,
        max_tokens=64,
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
    )

    text = ""
    for block in getattr(resp, "content", []) or []:
        if getattr(block, "type", None) == "text":
            text += getattr(block, "text", "")

    print(text.strip() or "<empty response>")


if __name__ == "__main__":
    main()

