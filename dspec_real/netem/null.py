"""Zero-delay emulator: isolates transport overhead from injected latency."""
from __future__ import annotations


class NullEmulator:
    name = "null"

    def one_way_ms(self, src: str, dst: str) -> float:
        return 0.0

    def install(self) -> None:
        pass
