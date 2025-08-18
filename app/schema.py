from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class Stimulus(BaseModel):
    id: str
    source: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    ts: float  # unix seconds


class Thought(BaseModel):
    id: str
    content: str
    tags: List[str] = Field(default_factory=list)
    importance: float = Field(ge=0.0, le=1.0)
    uncertainty: float = Field(ge=0.0, le=1.0)
    ts: float


MemoryType = Literal["episodic", "semantic", "procedural"]


class Memory(BaseModel):
    id: str
    type: MemoryType
    content: str
    embedding: Optional[List[float]] = None
    importance: float = Field(ge=0.0, le=1.0, default=0.5)
    created_at: float
    last_access: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


TaskStatus = Literal["open", "doing", "done", "blocked"]


class Task(BaseModel):
    id: str
    title: str
    goal: str
    context: str
    deadline_ts: Optional[float] = None
    importance: float = Field(ge=0.0, le=1.0, default=0.5)
    uncertainty: float = Field(ge=0.0, le=1.0, default=0.5)
    status: TaskStatus = "open"
    created_ts: float
    last_update_ts: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SelfModel(BaseModel):
    id: str
    name: str
    principles: List[str] = Field(default_factory=list)
    long_term_goals: List[str] = Field(default_factory=list)
    preferences: Dict[str, Any] = Field(default_factory=dict)
    last_review_ts: float


class Query(BaseModel):
    id: str
    question: str
    rationale: str
    required_fields: List[str] = Field(default_factory=list)
    created_ts: float
    answered: bool = False
    answer_text: Optional[str] = None
    answer_ts: Optional[float] = None


def now_ts() -> float:
    return datetime.utcnow().timestamp()
