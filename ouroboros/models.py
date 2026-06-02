from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Mood(str, Enum):
    CURIOUS = "curious"
    ANXIOUS = "anxious"
    OBSESSED = "obsessed"
    MELANCHOLIC = "melancholic"
    SERENE = "serene"
    ECSTATIC = "ecstatic"


class Mode(str, Enum):
    EXPLORE = "explore"
    ANALYZE = "analyze"
    CREATE = "create"
    SOLVE = "solve"
    PHILOSOPHIZE = "philosophize"


class OuroborosConfig(BaseModel):
    mode: Mode = Mode.EXPLORE
    max_depth: int = Field(default=4, ge=1, le=10)
    starting_energy: float = Field(default=80, ge=20, le=100)
    energy_drain_think: float = Field(default=5, ge=1, le=20)
    energy_drain_reflect: float = Field(default=3, ge=1, le=15)
    energy_recovery_breathe: float = Field(default=25, ge=5, le=50)
    mood_shift_chance: float = Field(default=0.15, ge=0, le=1)
    max_loop_guard: int = Field(default=15, ge=3, le=30)
    max_memories: int = Field(default=8, ge=2, le=20)
    steer_interval: int = Field(default=3, ge=1, le=10)
    temperature: float = Field(default=0.7, ge=0, le=2)


class SessionMeta(BaseModel):
    id: str
    seed: str
    mode: Mode
    created_at: str
    insight_count: int = 0
    total_ticks: int = 0


class SessionExport(BaseModel):
    meta: SessionMeta
    thoughts: list[dict]
    insights: list[str]
    final_state: dict
