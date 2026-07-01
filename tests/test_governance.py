"""Governance + prompt-registry unit tests — no DB, no Langfuse, no Ollama.

Proves the fallback contracts that keep agent-hub working air-gapped:
  * PromptRegistry returns bundled defaults when Langfuse is disabled.
  * GovernanceSettings enforces bounds (bad values rejected before a graph).
  * defaults_from_settings round-trips the shipped config into the knob set.
"""

import pytest
from pydantic import ValidationError

from agent_hub.adapters.governance_repo import defaults_from_settings
from agent_hub.adapters.prompts import DEFAULT_PROMPTS, PromptRegistry
from agent_hub.config import Settings
from agent_hub.domain.models import GovernanceSettings


def test_prompt_registry_falls_back_to_defaults_when_langfuse_off():
    reg = PromptRegistry(Settings(langfuse_enabled=False))
    for key, text in DEFAULT_PROMPTS.items():
        assert reg.get(key) == text


def test_prompt_registry_caches():
    reg = PromptRegistry(Settings(langfuse_enabled=False))
    first = reg.get("crag/grade")
    assert reg.get("crag/grade") is first  # served from cache


def test_defaults_from_settings_matches_config():
    s = Settings()
    gov = defaults_from_settings(s)
    assert gov.top_k == s.crag_top_k
    assert gov.cross_encoder_enabled == s.cross_encoder_enabled
    assert gov.generate_temperature == s.crag_generate_temperature
    assert gov.max_loops == s.crag_max_loops


def test_governance_rejects_out_of_range():
    base = defaults_from_settings(Settings()).model_dump()
    with pytest.raises(ValidationError):
        GovernanceSettings(**{**base, "top_k": 0})          # below min
    with pytest.raises(ValidationError):
        GovernanceSettings(**{**base, "generate_temperature": 9.0})  # above max
    with pytest.raises(ValidationError):
        GovernanceSettings(**{**base, "min_score": 1.5})    # above max


def test_governance_rejects_unknown_field():
    base = defaults_from_settings(Settings()).model_dump()
    with pytest.raises(ValidationError):
        GovernanceSettings(**{**base, "bogus": 1})


if __name__ == "__main__":
    test_prompt_registry_falls_back_to_defaults_when_langfuse_off(); print("prompt_fallback   OK")
    test_prompt_registry_caches();                                   print("prompt_cache      OK")
    test_defaults_from_settings_matches_config();                    print("defaults_roundtrip OK")
    test_governance_rejects_out_of_range();                          print("bounds            OK")
    test_governance_rejects_unknown_field();                         print("unknown_field     OK")
    print("\nALL GOVERNANCE TESTS PASSED")
