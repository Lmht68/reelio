import pytest

from src.transcript.cleaner import clean_transcript


class TestCleanTranscript:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            # Basic single fillers
            ("um, so I was thinking", "so I was thinking"),
            ("uh, I don't know", "I don't know"),
            ("ah, that makes sense", "that makes sense"),
            ("er, what was I saying", "what was I saying"),
            ("erm, let me check", "let me check"),
            ("hmm, interesting", "interesting"),
            ("huh, what did you say", "what did you say"),
            ("mm, I agree", "I agree"),
            # Elongated variants
            ("ummm, well maybe", "well maybe"),
            ("uhhh, not sure", "not sure"),
            ("ahhh I see", "I see"),
            ("errr, I forgot", "I forgot"),
            ("ermmm, the thing is", "the thing is"),
            ("hmmm, let me think", "let me think"),
            ("huhhh, that's weird", "that's weird"),
            ("mmmm, delicious", "delicious"),
            # Hyphenated / combined fillers
            ("uh-huh, I understand", "I understand"),
            ("mm-hmm, that's right", "that's right"),
            ("uh huh, got it", "got it"),
            ("mm hmm, exactly", "exactly"),
            ("uhhuh, sure", "sure"),
            ("mmhmm, yep", "yep"),
            # Fillers mid-sentence
            ("It's like, uh, interesting", "It's like, interesting"),
            ("And then, er, we went home", "And then, we went home"),
            ("That's, huh, strange", "That's, strange"),
            # Multiple fillers
            ("Um, uh, well, I guess", "well, I guess"),
            ("So um, like uh, you know", "So like you know"),
            ("ah er um, let's go", "let's go"),
            # Filler at start / end
            ("Um, I like it", "I like it"),
            ("I like it, um", "I like it,"),
            # Case insensitivity
            ("UM, hello", "hello"),
            ("Uh, Uh-Huh, ER", ""),
            ("uM, sO UmM", "sO"),  # both "uM" and "UmM" are fillers
            # Punctuation variants
            ("um... what next", "what next"),
            ("uh! that's it", "that's it"),
            ("er? really", "really"),
            ("hmm; maybe", "maybe"),
            ("ah: the thing", "the thing"),
            # No fillers — text preserved
            ("Hello world", "Hello world"),
            ("This is a normal sentence.", "This is a normal sentence."),
            # False positives: real words containing filler substrings
            ("umbrella term human", "umbrella term human"),
            ("Ahab went to the Bahamas", "Ahab went to the Bahamas"),
            ("The ermine is an animal", "The ermine is an animal"),
            ("hammer hummer summer", "hammer hummer summer"),
            # Empty / whitespace
            ("", ""),
            ("   ", ""),
            # All fillers
            ("um, uh, er", ""),
            ("uh-huh mm-hmm", ""),
        ],
    )
    def test_clean_transcript(self, text: str, expected: str) -> None:
        assert clean_transcript(text) == expected

    def test_whitespace_normalized_after_removal(self) -> None:
        """Multiple filler removals should not leave double spaces."""
        result = clean_transcript("So,   um,   uh,   er,   yeah")
        assert result == "So, yeah"
