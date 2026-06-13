from __future__ import annotations

import importlib

import pytest


def _reload_settings() -> object:
    import grounded_doc_agent.config.settings as settings_module

    return importlib.reload(settings_module)


@pytest.fixture(autouse=True)
def restore_settings_module(monkeypatch: pytest.MonkeyPatch) -> None:
    yield
    monkeypatch.delenv("GROUNDED_GEMINI_MODEL", raising=False)
    monkeypatch.delenv("GROUNDED_ADK_GEMINI_MODEL", raising=False)
    _reload_settings()


def test_gemini_model_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROUNDED_GEMINI_MODEL", raising=False)
    monkeypatch.delenv("GROUNDED_ADK_GEMINI_MODEL", raising=False)
    settings = _reload_settings()

    assert settings.GEMINI_MODEL == "gemini-3.1-flash-lite"
    assert settings.ADK_GEMINI_MODEL == "gemini-3.1-flash-lite"


def test_adk_gemini_model_inherits_synthesis_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROUNDED_GEMINI_MODEL", "custom-synth")
    monkeypatch.delenv("GROUNDED_ADK_GEMINI_MODEL", raising=False)
    settings = _reload_settings()

    assert settings.GEMINI_MODEL == "custom-synth"
    assert settings.ADK_GEMINI_MODEL == "custom-synth"


def test_adk_gemini_model_uses_override_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROUNDED_GEMINI_MODEL", "custom-synth")
    monkeypatch.setenv("GROUNDED_ADK_GEMINI_MODEL", "custom-adk")
    settings = _reload_settings()

    assert settings.GEMINI_MODEL == "custom-synth"
    assert settings.ADK_GEMINI_MODEL == "custom-adk"
