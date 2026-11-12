
import re

from analysis.skill_extractor import SkillMention


# Different names that refer to the same skill
SKILL_ALIASES = {
    "js": "JavaScript",
    "javascript": "JavaScript",

    "py": "Python",

    "ml": "Machine Learning",
    "machine learning": "Machine Learning",

    "react.js": "React",
    "reactjs": "React",

    "postgres": "PostgreSQL",
    "postgresql": "PostgreSQL",

    "node.js": "Node.js",
    "nodejs": "Node.js",

    "ts": "TypeScript",
    "typescript": "TypeScript",

    "aws": "Amazon Web Services",
}


class SkillNormalizer:

    @staticmethod
    def _clean_name(skill_name: str) -> str:
        """
        Clean a skill name before checking its aliases.
        """

        cleaned = skill_name.strip().lower()

        # Replace repeated whitespace with a single space
        cleaned = re.sub(r"\s+", " ", cleaned)

        return cleaned

    @classmethod
    def normalize_skill_name(cls, skill_name: str) -> str:
        """
        Convert a skill name into its canonical name.
        """

        cleaned_name = cls._clean_name(skill_name)

        if not cleaned_name:
            return ""

        return SKILL_ALIASES.get(
            cleaned_name,
            skill_name.strip()
        )

    @classmethod
    def normalize_skills(
        cls,
        skills: list[SkillMention]
    ) -> list[SkillMention]:
        """
        Normalize skill names and merge duplicate mentions.

        Skills are considered duplicates when they have the same
        normalized name and the same source.
        """

        normalized_skills = {}

        for skill in skills:

            normalized_name = cls.normalize_skill_name(
                skill.name
            )

            if not normalized_name:
                continue

            key = (normalized_name.lower(), skill.source)

            if key not in normalized_skills:
                normalized_skills[key] = skill.model_copy(
                    update={
                        "name": normalized_name,
                        "evidence": list(skill.evidence)
                    }
                )

            else:
                existing_skill = normalized_skills[key]

                # Keep all evidence from duplicate mentions
                existing_skill.evidence = list(
                    dict.fromkeys(
                        existing_skill.evidence + skill.evidence
                    )
                )

                # Keep the strongest available job metadata
                if existing_skill.importance is None:
                    existing_skill.importance = skill.importance

                if existing_skill.expected_level is None:
                    existing_skill.expected_level = (
                        skill.expected_level
                    )

        return list(normalized_skills.values())
