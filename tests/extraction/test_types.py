"""Screen Work identity primitive contract tests."""

from datetime import date

from reelio.extraction.types import (
    MINIMUM_SCREEN_WORK_MENTION_YEAR,
    maximum_screen_work_mention_year,
    normalize_screen_work_title,
)


def test_normalize_screen_work_title_canonicalizes_unicode_and_whitespace() -> None:
    """Normalize equivalent Screen Work title spellings to one identity."""
    assert (
        normalize_screen_work_title("  AME\u0301LIE:\tLe Fabuleux\nDestin!  ")
        == "AMÉLIE: Le Fabuleux Destin!"
    )


def test_screen_work_mention_year_policy() -> None:
    """Allow Screen Work Mention years from 1888 through two future years."""
    assert MINIMUM_SCREEN_WORK_MENTION_YEAR == 1888
    assert maximum_screen_work_mention_year() == date.today().year + 2
