from __future__ import annotations

from urllib.parse import urlparse
from urllib import robotparser


class RobotsGate:
    """
    Domain allow/deny with robots.txt compliance.
    - allow_domains can contain "*" to allow all domains.
    - file:// URLs are always allowed (used for offline tests).
    """
    def __init__(self, allow_domains: list[str]) -> None:
        self.allow = set(allow_domains)
        self.cache: dict[str, robotparser.RobotFileParser] = {}

    def _rp(self, netloc: str) -> robotparser.RobotFileParser:
        if netloc not in self.cache:
            rp = robotparser.RobotFileParser()
            # Prefer https; most sites will redirect if needed
            rp.set_url(f"https://{netloc}/robots.txt")
            try:
                rp.read()
            except Exception:
                # Fail-safe: deny if robots cannot be read
                # (RobotFileParser has no official disallow_all, mimic by can_fetch always False)
                class _Deny(robotparser.RobotFileParser):  # type: ignore[misc]
                    def can_fetch(self_inner, *_args, **_kwargs) -> bool:  # noqa: N802
                        return False
                rp = _Deny()  # type: ignore[assignment]
            self.cache[netloc] = rp
        return self.cache[netloc]

    def allowed(self, url: str) -> bool:
        # Allow local file fixtures
        if url.startswith("file://"):
            return True

        netloc = urlparse(url).netloc
        if not netloc:
            return False

        # Wildcard: allow any domain, but still respect robots.txt
        if "*" in self.allow:
            return self._rp(netloc).can_fetch("GhostScraper", url)

        # Otherwise only domains in allowlist
        if netloc not in self.allow:
            return False

        return self._rp(netloc).can_fetch("GhostScraper", url)
