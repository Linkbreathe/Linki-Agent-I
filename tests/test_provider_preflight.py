from __future__ import annotations

import pytest

from Linki.providers.openai_provider import required_env_for_provider, validate_provider_config


def test_validate_provider_config_requires_selected_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY is not set"):
        validate_provider_config("openai")


def test_validate_provider_config_accepts_present_key(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    validate_provider_config("deepseek")


def test_required_env_for_provider_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported provider"):
        required_env_for_provider("bogus")
