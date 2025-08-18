from __future__ import annotations

from urllib.parse import urlparse
from urllib import robotparser


class RobotsGate:
    def __init__(self, allow_domains: list[str]) -> None:
        self.allow = set(allow_domains)
        self.cache: dict[str, robotparser.RobotFileParser] = {}

    def _rp(self, netloc: str) -> robotparser.RobotFileParser:
        if netloc not in self.cache:
            rp = robotparser.RobotFileParser()
            rp.set_url(f"https://{netloc}/robots.txt")
            try:
                rp.read()
            except Exception:
                # Be safe: deny if cannot read
                rp.disallow_all = True  # type: ignore[attr-defined]
            self.cache[netloc] = rp
        return self.cache[netloc]

    def allowed(self, url: str) -> bool:
        netloc = urlparse(url).netloc
        if netloc and netloc not in self.allow:
            return False
        if url.startswith("file://"):
            return True
        rp = self._rp(netloc)
        return rp.can_fetch("GhostScraper", url)
