from __future__ import annotations

from pathlib import Path

import pytest

from paperdaily.config import ConfigError, PaperDailyConfig, load_config, save_config
from paperdaily.topics import Topic, TopicMatcher


def embodied_topic(**overrides):
    values = {
        "id": "embodied-vla",
        "name": "具身智能与 VLA",
        "description": "VLA, WAM, robot learning",
        "arxiv_categories": ["cs.RO", "cs.AI", "cs.CV"],
        "exact_phrases": ["vision-language-action", "world action model"],
        "keywords": ["VLA", "WAM", "robotic manipulation"],
        "context_keywords": ["robot", "robotic", "manipulation", "embodied", "policy"],
        "negative_keywords": ["wireless access management", "web application monitoring"],
        "daily_limit": 12,
        "minimum_score": 0.25,
    }
    values.update(overrides)
    return Topic(**values)


def test_load_config_supports_nested_user_runtime_and_topics(tmp_path: Path) -> None:
    path = tmp_path / "paperdaily.yaml"
    path.write_text(
        """
user:
  id: alice
  timezone: Asia/Shanghai
runtime:
  output_dir: output
  database: state/paperdaily.db
daily:
  limit: 10
  rerank_limit: 25
  llm_rerank_enabled: false
  llm_rerank_weight: 0.4
  llm_rerank_input_cost_per_million_tokens: 2.5
  llm_rerank_output_cost_per_million_tokens: 9.0
catchup:
  auto_catchup_days: 2
  first_run_days: 7
providers:
  default_provider: auto
  fallback_order: [codex, claude]
  codex:
    enabled: true
deep_read:
  isolated_home: true
topics:
  - id: embodied-vla
    name: Embodied VLA
    categories: [cs.RO, cs.AI]
    phrases: [vision-language-action]
    keywords: [VLA]
    context: [robot]
    negative_keywords: [wireless access management]
    daily_limit: 8
    minimum_score: 0.3
""".strip(),
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.user_id == "alice"
    assert config.timezone == "Asia/Shanghai"
    assert config.output_dir == tmp_path / "output"
    assert config.database == tmp_path / "state" / "paperdaily.db"
    assert config.daily.default_limit == 10
    assert config.daily.limit == 10
    assert config.daily.llm_rerank_enabled is False
    assert config.daily.llm_rerank_weight == 0.4
    assert config.daily.llm_rerank_input_cost_per_million_tokens == 2.5
    assert config.daily.llm_rerank_output_cost_per_million_tokens == 9.0
    assert config.catchup.first_run_days == 7
    assert config.providers.fallback_order == ["codex", "claude"]
    assert config.providers.settings["codex"]["enabled"] is True
    assert config.deep_read.isolated_home is True
    assert config.topics[0].arxiv_categories == ["cs.RO", "cs.AI"]
    assert config.topics[0].exact_phrases == ["vision-language-action"]


def test_config_round_trip_preserves_core_values(tmp_path: Path) -> None:
    config = PaperDailyConfig(
        user_id="researcher",
        output_dir=tmp_path / "out",
        database=tmp_path / "state.db",
        topics=[embodied_topic()],
    )
    path = save_config(config, tmp_path / "config.yaml")
    loaded = load_config(path)

    assert loaded.user_id == config.user_id
    assert loaded.output_dir == config.output_dir
    assert loaded.database == config.database
    assert loaded.topics[0].to_dict() == config.topics[0].to_dict()
    serialized = config.to_dict()
    assert "providers" not in serialized
    assert "deep_read" not in serialized


def test_zero_daily_limits_mean_no_cap() -> None:
    config = PaperDailyConfig.from_dict({"daily": {"default_limit": 0}})
    topic = embodied_topic(daily_limit=0)

    assert config.daily.default_limit == 0
    assert topic.daily_limit == 0


def test_config_rejects_empty_timezone_and_duplicate_topics() -> None:
    with pytest.raises(ConfigError, match="timezone is required"):
        PaperDailyConfig(timezone="")

    with pytest.raises(ConfigError, match="duplicate topic"):
        PaperDailyConfig(topics=[embodied_topic(), embodied_topic()])


def test_deep_read_isolated_home_requires_a_boolean() -> None:
    with pytest.raises(ConfigError, match="deep_read.isolated_home must be a boolean"):
        PaperDailyConfig.from_dict({"deep_read": {"isolated_home": "true"}})


def test_deep_read_pdf_resource_limits_are_validated_and_serialized() -> None:
    config = PaperDailyConfig.from_dict(
        {
            "deep_read": {
                "max_pdf_pages": 123,
                "max_extracted_text_chars": 456_789,
            }
        }
    )

    assert config.deep_read.max_pdf_pages == 123
    assert config.deep_read.max_extracted_text_chars == 456_789
    assert config.deep_read.to_dict()["max_pdf_pages"] == 123

    for field, value in (("max_pdf_pages", 0), ("max_extracted_text_chars", True)):
        with pytest.raises(ConfigError, match=f"deep_read.{field} must be a positive integer"):
            PaperDailyConfig.from_dict({"deep_read": {field: value}})


def test_title_match_scores_higher_than_same_abstract_match() -> None:
    matcher = TopicMatcher([embodied_topic(keywords=["robotic manipulation"], exact_phrases=[])])

    title_match = matcher.match("Robotic Manipulation with Generalist Policies", "A benchmark.")
    abstract_match = matcher.match("A Generalist Benchmark", "We study robotic manipulation.")

    assert title_match.matched
    assert abstract_match.matched
    assert title_match.score > abstract_match.score


@pytest.mark.parametrize("acronym", ["VLA", "WAM"])
def test_ambiguous_acronym_requires_context(acronym: str) -> None:
    matcher = TopicMatcher([embodied_topic(exact_phrases=[], minimum_score=0.2)])

    assert not matcher.match(f"A new {acronym} architecture", "General systems study.").matched
    contextual = matcher.match(f"A new {acronym} architecture", "A robot manipulation policy.")
    assert contextual.matched
    assert acronym in contextual.matched_terms


def test_expanded_exact_phrase_matches_hyphen_or_space_without_context() -> None:
    matcher = TopicMatcher([embodied_topic(keywords=[], context_keywords=[], minimum_score=0.4)])

    spaced = matcher.match("Scaling Vision Language Action Models", "")
    hyphenated = matcher.match("Scaling Vision-Language-Action Models", "")

    assert spaced.matched and hyphenated.matched
    assert spaced.topics == ["embodied-vla"]
    assert "vision-language-action" in spaced.matched_terms


@pytest.mark.parametrize(
    ("acronym", "expanded_title"),
    [
        ("VLA", "Vision-Language-Action Models for Robot Control"),
        ("WAM", "World Action Model for Robotic Manipulation"),
    ],
)
def test_keyword_only_acronym_topics_also_match_their_expanded_form(
    acronym: str,
    expanded_title: str,
) -> None:
    matcher = TopicMatcher([
        embodied_topic(
            exact_phrases=[],
            keywords=[acronym],
            context_keywords=[],
            minimum_score=0.4,
        )
    ])

    result = matcher.match(expanded_title, "")

    assert result.matched
    assert result.matched_terms


def test_negative_keyword_blocks_topic_even_when_acronym_and_context_match() -> None:
    matcher = TopicMatcher([embodied_topic()])
    result = matcher.match(
        "VLA for wireless access management",
        "A policy for network control rather than robotics.",
    )

    assert not result.matched
    assert result.score == 0.0
    assert "wireless access management" in result.negative_terms


def test_annotate_and_filter_return_explainable_fields() -> None:
    matcher = TopicMatcher([embodied_topic()])
    papers = [
        {"title": "World-Action Model for Robot Policies", "abstract": "Embodied control."},
        {"title": "A Database Index", "abstract": "Query optimization."},
    ]

    selected = matcher.filter(papers)

    assert len(selected) == 1
    assert selected[0]["matched_topics"] == ["embodied-vla"]
    assert selected[0]["matched_terms"]
    assert selected[0]["topic_score"] > 0
    assert selected[0]["topic_match"]["matched"] is True
