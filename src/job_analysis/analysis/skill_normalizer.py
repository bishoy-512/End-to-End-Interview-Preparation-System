import json
import re
from functools import lru_cache
from pathlib import Path

from analysis.skill_extractor import SkillMention

TAXONOMY_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "skills_taxonomy.json"
)

_IMPORTANCE_RANK = {"required": 2, "preferred": 1}


@lru_cache(maxsize=1)
def load_taxonomy() -> dict[str, dict]:
    """Canonical skill -> {category, aliases, related}."""
    with TAXONOMY_PATH.open(encoding="utf-8") as file:
        return json.load(file)["skills"]


def category_of(skill_name: str) -> str | None:
    info = load_taxonomy().get(skill_name)
    return info["category"] if info else None


def related_skills(skill_name: str) -> list[str]:
    info = load_taxonomy().get(skill_name)
    return list(info["related"]) if info else []


@lru_cache(maxsize=1)
def _exact_aliases() -> dict[str, str]:
    table = {}
    for canonical, info in load_taxonomy().items():
        table[canonical.lower()] = canonical
        for alias in info["aliases"]:
            table[alias.lower()] = canonical
    return table


@lru_cache(maxsize=1)
def _alias_patterns() -> list[tuple[re.Pattern, str]]:
    """Whole-word alias patterns, longest alias first ('machine learning' before 'ml')."""
    pairs = []
    for alias, canonical in _exact_aliases().items():
        pattern = re.compile(
            rf"(?<![a-z0-9+#.]){re.escape(alias)}(?![a-z0-9+#])"
        )
        pairs.append((len(alias), pattern, canonical))
    pairs.sort(key=lambda item: item[0], reverse=True)
    return [(pattern, canonical) for _, pattern, canonical in pairs]


# Kept for backward compatibility with code that imports it.
SKILL_ALIASES = dict(_exact_aliases())


class SkillNormalizer:

    @staticmethod
    def _clean_name(skill_name: str) -> str:
        return re.sub(r"\s+", " ", skill_name.strip().lower())

    @classmethod
    def normalize_skill_names(cls, skill_name: str) -> list[str]:
        """
        Return the canonical skill names found in a raw skill name.

        "Python programming skills" -> ["Python"]
        "NumPy and Pandas"          -> ["NumPy", "Pandas"]
        Unknown skills are kept as written, so nothing is dropped.
        """
        cleaned = cls._clean_name(skill_name)
        if not cleaned:
            return []

        exact = _exact_aliases().get(cleaned)
        if exact:
            return [exact]

        remaining = cleaned
        found: list[tuple[int, str]] = []
        for pattern, canonical in _alias_patterns():
            match = pattern.search(remaining)
            if match:
                found.append((match.start(), canonical))
                # Blank out the match so shorter aliases cannot reuse it.
                remaining = (
                    remaining[: match.start()]
                    + " " * (match.end() - match.start())
                    + remaining[match.end():]
                )

        if not found:
            return [skill_name.strip()]

        names = []
        for _, canonical in sorted(found):
            if canonical not in names:
                names.append(canonical)
        return names

    @classmethod
    def normalize_skill_name(cls, skill_name: str) -> str:
        """First canonical name for a raw skill name ('' when empty)."""
        names = cls.normalize_skill_names(skill_name)
        return names[0] if names else ""

    @classmethod
    def normalize_skills(
        cls,
        skills: list[SkillMention]
    ) -> list[SkillMention]:
        """
        Normalize skill names and merge duplicate mentions.

        A raw name that holds several skills becomes several mentions.
        Mentions are duplicates when they share the canonical name and source.
        The strongest importance (required over preferred) is kept.
        """
        merged: dict[tuple[str, str], SkillMention] = {}

        for skill in skills:
            for name in cls.normalize_skill_names(skill.name):
                key = (name.lower(), skill.source)

                if key not in merged:
                    merged[key] = skill.model_copy(
                        update={
                            "name": name,
                            "evidence": list(skill.evidence),
                        }
                    )
                    continue

                existing = merged[key]
                existing.evidence = list(
                    dict.fromkeys(existing.evidence + skill.evidence)
                )

                if _IMPORTANCE_RANK.get(skill.importance, 0) > _IMPORTANCE_RANK.get(existing.importance, 0):
                    existing.importance = skill.importance

                if existing.expected_level is None:
                    existing.expected_level = skill.expected_level

        return list(merged.values())
