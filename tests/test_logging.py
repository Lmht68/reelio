"""Behavioral tests for structured logging configuration."""

import io
import logging

from reelio.logging import configure_logging


def test_log_level_controls_reelio_namespace() -> None:
    """Configure the reelio namespace at the requested severity."""
    logger = logging.getLogger("reelio")

    configure_logging("DEBUG")
    assert logger.getEffectiveLevel() == logging.DEBUG

    configure_logging("INFO")
    assert not logger.isEnabledFor(logging.DEBUG)


def test_extra_fields_are_rendered_as_key_value_pairs() -> None:
    """Render structured extra fields in the configured stream output."""
    configure_logging("INFO")
    logger = logging.getLogger("reelio.test")
    handler = logging.getLogger("reelio").handlers[0]
    assert isinstance(handler, logging.StreamHandler)
    stream = io.StringIO()
    previous_stream = handler.setStream(stream)

    try:
        logger.info("transcript acquired", extra={"stage": "transcription"})
        handler.flush()
    finally:
        handler.setStream(previous_stream)

    assert "transcript acquired" in stream.getvalue()
    assert "stage=transcription" in stream.getvalue()
    assert "message=transcript acquired" not in stream.getvalue()
