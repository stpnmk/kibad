"""
tests/test_i18n.py – Unit tests for core/i18n.py
"""
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.i18n import t, available_keys, available_languages, register


# ---------------------------------------------------------------------------
# t() — basic translation
# ---------------------------------------------------------------------------

def test_t_returns_russian_string():
    result = t("app_title", lang="ru")
    assert isinstance(result, str)
    assert len(result) > 0
    assert "KIBAD" in result


def test_t_returns_english_string():
    result = t("app_title", lang="en")
    assert isinstance(result, str)
    assert "KIBAD" in result


def test_t_russian_and_english_differ():
    ru = t("nav_data", lang="ru")
    en = t("nav_data", lang="en")
    assert ru != en


def test_t_known_key_russian():
    result = t("select_dataset", lang="ru")
    assert isinstance(result, str)
    assert len(result) > 0


def test_t_known_key_english():
    result = t("select_dataset", lang="en")
    assert result == "Select dataset"


# ---------------------------------------------------------------------------
# t() — format kwargs
# ---------------------------------------------------------------------------

def test_t_with_format_kwargs():
    result = t("warn_few_observations", lang="en", n=5)
    assert "5" in result


def test_t_with_format_kwargs_russian():
    result = t("warn_few_observations", lang="ru", n=10)
    assert "10" in result


def test_t_format_kwargs_missing_key_no_crash():
    # If format key is missing from kwargs, should not crash
    result = t("warn_few_observations", lang="en")
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# t() — unknown keys
# ---------------------------------------------------------------------------

def test_t_unknown_key_returns_key():
    result = t("this_key_does_not_exist_xyz", lang="en")
    assert result == "this_key_does_not_exist_xyz"


def test_t_unknown_key_returns_key_russian():
    result = t("nonexistent_key_abc", lang="ru")
    assert result == "nonexistent_key_abc"


# ---------------------------------------------------------------------------
# available_keys()
# ---------------------------------------------------------------------------

def test_available_keys_returns_non_empty_list():
    keys = available_keys()
    assert isinstance(keys, list)
    assert len(keys) > 0


def test_available_keys_contains_known_keys():
    keys = available_keys()
    assert "app_title" in keys
    assert "nav_data" in keys
    assert "select_dataset" in keys


# ---------------------------------------------------------------------------
# available_languages()
# ---------------------------------------------------------------------------

def test_available_languages_returns_ru_en():
    langs = available_languages()
    assert langs == ["ru", "en"]


def test_available_languages_returns_list():
    langs = available_languages()
    assert isinstance(langs, list)
    assert len(langs) == 2


# ---------------------------------------------------------------------------
# register()
# ---------------------------------------------------------------------------

def test_register_adds_new_key():
    register("_test_custom_key", {"ru": "Тест", "en": "Test"})
    assert "_test_custom_key" in available_keys()
    assert t("_test_custom_key", lang="en") == "Test"
    assert t("_test_custom_key", lang="ru") == "Тест"


def test_register_overwrites_existing_key():
    register("_test_overwrite", {"ru": "Первый", "en": "First"})
    assert t("_test_overwrite", lang="en") == "First"
    register("_test_overwrite", {"ru": "Второй", "en": "Second"})
    assert t("_test_overwrite", lang="en") == "Second"


def test_register_with_format_placeholders():
    register("_test_fmt", {"ru": "Значение: {val}", "en": "Value: {val}"})
    result = t("_test_fmt", lang="en", val=42)
    assert result == "Value: 42"
