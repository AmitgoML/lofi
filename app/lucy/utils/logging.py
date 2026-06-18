import os
import json
import logging
import sys
from loguru import logger


class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except Exception:
            level = record.levelno
        message_text = record.getMessage()
        if isinstance(message_text, str) and (
            "<" in message_text or ">" in message_text
        ):
            message_text = message_text.replace("<", "\\<").replace(">", "\\>")
        logger.bind(logger_name=record.name).opt(
            depth=6, exception=record.exc_info, colors=True
        ).log(level, message_text)


def setup_logging(level: str = "INFO") -> None:
    logger.remove()
    env_name = (os.getenv("APP_ENV") or "production").strip().lower()
    env_level = (os.getenv("LOG_LEVEL") or "").strip().upper()
    effective_level = env_level or ("DEBUG" if env_name == "local" else level)

    if env_name != "local":
        # CloudWatch-friendly minimal JSON logs in non-local envs
        def _json_sink(message):
            rec = message.record
            payload = {
                "ts": rec["time"].isoformat(),
                "level": rec["level"].name,
                "msg": rec["message"],
            }
            sys.stdout.write(json.dumps(payload) + "\n")
            sys.stdout.flush()

        logger.add(
            _json_sink,
            backtrace=False,
            diagnose=False,
            enqueue=False,  # ← changed
            level=effective_level,
        )
    else:
        # Human-friendly pretty logs for local development
        def _escape_message_filter(record):
            try:
                msg = record.get("message", "")
                if isinstance(msg, str) and ("<" in msg or ">" in msg):
                    record["message"] = msg.replace("<", "\\<").replace(">", "\\>")
            except Exception:
                pass
            return True

        logger.add(
            sys.stdout,
            colorize=True,
            backtrace=True,
            diagnose=False,
            enqueue=False,  # ← changed
            level=effective_level,
            format=(
                "<green>{time:HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            ),
            filter=_escape_message_filter,
        )

    # Route stdlib logging (uvicorn/fastapi) into loguru
    targets = ["uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"]
    for name in targets:
        logging.getLogger(name).handlers = [InterceptHandler()]
        logging.getLogger(name).propagate = False

    logging.basicConfig(
        handlers=[InterceptHandler()],
        level=getattr(logging, (effective_level or "INFO").upper(), logging.INFO),
    )
