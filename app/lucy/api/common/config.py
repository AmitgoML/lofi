from __future__ import annotations

import os

# Default agent used when no agent is specified
DEFAULT_AGENT_NAME = "lucy_generic"
# Fallback session id when none is provided
DEFAULT_SESSION_ID = "default"
# Number of recent messages to include in context (increased for better follow-up context)
RECENT_MSGS_LIMIT = int(os.getenv("RECENT_MSGS_LIMIT", "20"))
# Seconds to debounce streamed deltas for smoother UI
STREAM_DEBOUNCE = 0.01
# Max total visible characters from history to send to the model (lower to reduce tokens)
HISTORY_CHAR_BUDGET = int(os.getenv("HISTORY_CHAR_BUDGET", "4000"))
# Max visible chars per individual history message (post-sanitization)
PER_MESSAGE_CHAR_LIMIT = int(os.getenv("PER_MESSAGE_CHAR_LIMIT", "600"))
# Seconds between heartbeat keepalives during tool execution (prevents idle timeouts)
HEARTBEAT_INTERVAL_S = int(os.getenv("HEARTBEAT_INTERVAL_S", "15"))
# Max total stream duration before timeout (safely under App Runner 120s limit)
MAX_STREAM_DURATION_S = int(os.getenv("MAX_STREAM_DURATION_S", "110"))
