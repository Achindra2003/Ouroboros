"""Fixed evaluation task set.

A frozen set of seed prompts spanning every mode. Keep this stable so results
are comparable across runs; add new tasks rather than editing existing ones.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    id: str
    seed: str
    mode: str


TASKS: list[Task] = [
    # explore
    Task("exp-1", "What is the relationship between attention and identity?", "explore"),
    Task("exp-2", "Why do we find some questions more beautiful than their answers?", "explore"),
    Task("exp-3", "What would change if memory were a choice rather than a given?", "explore"),
    # analyze
    Task("ana-1", "A startup assumes 40% month-over-month growth for two years. What are the blind spots?", "analyze"),
    Task("ana-2", "Remote-first teams report higher satisfaction but slower onboarding. What explains the tension?", "analyze"),
    Task("ana-3", "Why might a metric that improves user engagement still harm the product long-term?", "analyze"),
    # create
    Task("cre-1", "Invent a new ritual for marking the end of a creative project.", "create"),
    Task("cre-2", "Design an object that helps two strangers trust each other faster.", "create"),
    Task("cre-3", "What is an unconventional use for the concept of 'silence'?", "create"),
    # solve
    Task("sol-1", "A team ships fast but accrues bugs faster. How do you break the cycle without halting delivery?", "solve"),
    Task("sol-2", "How do you keep a knowledge base accurate when no one is incentivized to maintain it?", "solve"),
    Task("sol-3", "Users churn silently after 3 weeks. How would you find out why and intervene?", "solve"),
    # philosophize
    Task("phi-1", "Is a contradiction you can live with wiser than a consistency you cannot?", "philosophize"),
    Task("phi-2", "If you could no longer remember your reasons, would your values still be yours?", "philosophize"),
    Task("phi-3", "Does meaning require an ending?", "philosophize"),
]


def get_tasks(limit: int | None = None) -> list[Task]:
    return TASKS[:limit] if limit else list(TASKS)
