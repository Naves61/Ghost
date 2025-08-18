from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class ScrapePlan:
    question: str
    seeds: List[str] = field(default_factory=list)
    max_pages: int = 5
    allow_domains: List[str] = field(default_factory=list)
