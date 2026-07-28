import re

# Speech fillers: standalone interjections and hesitation sounds that carry no semantic content.
# Each pattern is anchored with \b to avoid matching substrings of real words (e.g. "um" in
# "umbrella", "er" in "term", "ah" in "Ahab"). Trailing punctuation (.,!?;:) is consumed so
# "um," and "uh..." are removed cleanly.
_FILLER_RE = re.compile(
    r"\b(?:"
    r"um+|"  # um, umm, ummm
    r"uh+(?:[- ]?huh)?|"  # uh, uhh, uh-huh, uh huh, uhhuh
    r"ah+|"  # ah, ahh, ahhh
    r"er+|"  # er, err, errr
    r"erm+|"  # erm, ermm, ermmm
    r"hmm+|"  # hmm, hmmm
    r"huh+|"  # huh, huhh
    r"mm+(?:[- ]?hmm?)?"  # mm, mmm, mm-hmm, mm hmm, mmhmm
    r")\b[.,!?;:]*\s*",
    re.IGNORECASE,
)


def clean_transcript(text: str) -> str:
    """Remove unambiguous speech fillers from transcript text.

    Removes fillers like um, uh, ah, er, erm, hmm, huh, mm, uh-huh, mm-hmm
    and their elongated variants (ummm, uhhh, etc.). Only standalone fillers
    are matched; fillers embedded in real words (e.g. "um" in "umbrella") are
    left untouched.

    Returns the cleaned text with whitespace normalized.
    """
    cleaned = _FILLER_RE.sub(" ", text)
    # Collapse runs of whitespace introduced by removals and strip ends.
    return re.sub(r"\s+", " ", cleaned).strip()
