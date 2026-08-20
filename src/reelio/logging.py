"""Configure Reelio's standard-library structured logging convention.

INFO records describe stage transitions, durations, and counts.
DEBUG records may include stage payloads, but never API keys, authorization
headers, or media bytes.
Structured fields use lowercase snake_case names and are passed through
``extra`` to the logger.
"""

import logging
import logging.config

_STANDARD_LOG_RECORD_FIELDS = frozenset(logging.makeLogRecord({}).__dict__)


class _KeyValueFormatter(logging.Formatter):
    """Append non-standard log record fields as key=value pairs."""

    def format(self, record: logging.LogRecord) -> str:
        """Format a record with its structured fields.

        Args:
            record: Log record to format.

        Returns:
            str: The rendered log message and structured fields.
        """
        fields = [
            f"{key}={record.__dict__[key]}"
            for key in sorted(record.__dict__)
            if key not in _STANDARD_LOG_RECORD_FIELDS and not key.startswith("_")
        ]
        message = super().format(record)
        return " ".join((message, *fields))


def configure_logging(level: str) -> None:
    """Configure the Reelio logger namespace and third-party log levels.

    Args:
        level: A standard logging level name such as ``INFO`` or ``DEBUG``.

    Returns:
        None: The logging configuration is applied globally.

    Raises:
        ValueError: If ``level`` is not recognized by the logging subsystem.
    """
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "key_value": {
                    "()": "reelio.logging._KeyValueFormatter",
                    "format": "%(levelname)s:   [%(name)s] %(message)s",
                }
            },
            "handlers": {
                "stderr": {
                    "class": "logging.StreamHandler",
                    "formatter": "key_value",
                    "stream": "ext://sys.stderr",
                }
            },
            "loggers": {
                "reelio": {
                    "handlers": ["stderr"],
                    "level": level.upper(),
                    "propagate": False,
                },
                "httpx": {"level": "WARNING"},
                "yt_dlp": {"level": "WARNING"},
                "faster_whisper": {"level": "WARNING"},
            },
            "root": {"handlers": ["stderr"], "level": "WARNING"},
        }
    )
