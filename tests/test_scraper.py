from __future__ import annotations

import tempfile
from pathlib import Path

from app.scraper.runner import Scraper
from app.scraper.plan import ScrapePlan
from app.settings import settings


def test_scraper_file_urls(tmp_path: Path):
    # create two linked files to simulate site
    p1 = tmp_path / "a.html"
    p2 = tmp_path / "b.html"
    p1.write_text(f"<html><body>Doc A <a href='file://{p2}'>B</a></body></html>", encoding="utf-8")
    p2.write_text("<html><body>Doc B content</body></html>", encoding="utf-8")

    plan = ScrapePlan(
        question="find docs",
        seeds=[f"file://{p1}"],
        max_pages=3,
        allow_domains=settings.allowlist(),
    )
    s = Scraper()
    saved = s.execute(plan)
    # Should have saved at least the first document
    assert saved
