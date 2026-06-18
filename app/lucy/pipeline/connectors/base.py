"""Abstract base for platform read connectors.

Implementations must enforce per-platform isolation and idempotent pulls.
Write-side connectors slot in via a separate interface in a future phase.
"""

from abc import ABC


class PlatformConnector(ABC):
    """Marker base for platform ingestion connectors. Implementation in P4-03+."""
