from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple
from urllib.parse import urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

from .plan import ScrapePlan
from .robots import RobotsGate
from ..settings import settings
from ..schema import Memory
from ..memory_longterm import upsert_memory
from ..providers.stubs import StubEmbeddings


def _normalize_url(u: str) -> str:
    p = urlparse(u)
    # remove fragments and params
    return urlunparse((p.scheme or "http", p.netloc, p.path, "", "", ""))


@dataclass
class Scraper:
    rate_last: Dict[str, float]

    def __init__(self) -> None:
        self.rate_last = {}

    def _rate_limit(self, netloc: str) -> None:
        wait = settings.SCRAPER_RATE_LIMIT_PER_DOMAIN
        last = self.rate_last.get(netloc, 0.0)
        now = time.time()
        if now - last < wait:
            time.sleep(wait - (now - last))
        self.rate_last[netloc] = time.time()

    def _fetch(self, url: str) -> Tuple[str, str]:
        if url.startswith("file://"):
            path = url[len("file://") :]
            with open(path, "r", encoding="utf-8") as f:
                return url, f.read()
        p = urlparse(url)
        self._rate_limit(p.netloc)
        headers = {"User-Agent": "GhostScraper/0.1 (+https://example.com)"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        return (url, resp.text)

    def execute(self, plan: ScrapePlan) -> List[str]:
        robots = RobotsGate(allow_domains=plan.allow_domains)
        seen: set[str] = set()
        to_visit: List[str] = [s for s in plan.seeds[: settings.SCRAPER_MAX_PAGES]]
        saved_ids: List[str] = []
        emb = StubEmbeddings()

        while to_visit and len(seen) < plan.max_pages:
            url = _normalize_url(to_visit.pop(0))
            if url in seen:
                continue
            seen.add(url)

            if not robots.allowed(url):
                continue
            try:
                raw_url, html = self._fetch(url)
            except Exception:
                continue

            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text(separator="\n", strip=True)
            content = f"{text[:4000]}"
            mem = Memory(
                id=f"doc:{int(time.time()*1000)}:{len(seen)}",
                type="semantic",
                content=f"[source:{raw_url}]\n{content}",
                embedding=emb.embed([content])[0],
                importance=0.2,
                created_at=time.time(),
                last_access=time.time(),
                metadata={"source": "scraper", "url": raw_url},
            )
            upsert_memory(mem)
            saved_ids.append(mem.id)

            # discover links (same domain only)
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                if href.startswith("#"):
                    continue
                if href.startswith("mailto:"):
                    continue
                if href.startswith("/"):
                    base = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
                    href = base + href
                n = _normalize_url(href)
                if robots.allowed(n) and urlparse(n).netloc == urlparse(url).netloc:
                    to_visit.append(n)

        return saved_ids
