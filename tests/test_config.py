import pytest
from pydantic import ValidationError

from src.config import Settings


class TestSettings:
    def test_secret_str_returns_raw_value(self) -> None:
        settings = Settings(_env_file=None, llm_api_key="sk-test-secret")
        assert settings.llm_api_key.get_secret_value() == "sk-test-secret"

    def test_repr_does_not_contain_secret(self) -> None:
        settings = Settings(_env_file=None, llm_api_key="sk-test-secret")
        assert "sk-test-secret" not in repr(settings)

    def test_str_does_not_contain_secret(self) -> None:
        settings = Settings(_env_file=None, llm_api_key="sk-test-secret")
        assert "sk-test-secret" not in str(settings.llm_api_key)

    def test_empty_key_defaults_to_empty(self) -> None:
        settings = Settings(_env_file=None)
        assert settings.llm_api_key.get_secret_value() == ""

    def test_entity_max_concurrent_default(self) -> None:
        settings = Settings(_env_file=None)
        assert settings.entity_max_concurrent == 4

    def test_entity_max_concurrent_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            Settings(_env_file=None, entity_max_concurrent=0)
