import pytest

from src.metadata.exceptions import MetadataError, MetadataProviderError


def test_inheritance():
    assert issubclass(MetadataProviderError, MetadataError)
    assert issubclass(MetadataError, Exception)


def test_metadata_provider_error_inherits_constructor():
    err = MetadataProviderError("safe message", original_error=ValueError("cause"))
    assert err.original_error is not None
    assert str(err) == "safe message"


def test_original_error_none_by_default():
    err = MetadataError("safe message")
    assert err.original_error is None


def test_original_error_keyword_only():
    with pytest.raises(TypeError):
        MetadataError("message", ValueError("cause"))  # type: ignore[misc]


def test_original_error_identity():
    cause = ValueError("root")
    err = MetadataError("safe", original_error=cause)
    assert err.original_error is cause


def test_str_equals_only_safe_message():
    err = MetadataError("safe message", original_error=ValueError("internal"))
    assert str(err) == "safe message"


def test_cause_text_not_leaked():
    secret = "sk-abc123secret"
    url = "https://api.example.com/v1/search?token=abc"
    body = '{"error": "invalid_grant"}'
    cause_text = f"failed call: {secret} url={url} body={body}"
    cause = ValueError(cause_text)
    err = MetadataError("provider lookup failed", original_error=cause)
    safe_str = str(err)
    assert secret not in safe_str
    assert url not in safe_str
    assert body not in safe_str
    assert "abc123secret" not in safe_str
    assert "invalid_grant" not in safe_str
    assert "token" not in safe_str
    assert "error" not in safe_str
    # The only token that should appear is the safe message itself
    assert safe_str == "provider lookup failed"
