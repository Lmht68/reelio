from unittest.mock import AsyncMock, MagicMock

import httpx
import openai
import pytest

from src.entities.exceptions import EntityExtractionError
from src.entities.extractors.llm import LLMEntityExtractor
from src.entities.models import EntityType


class TestLLMEntityExtractor:
    @pytest.fixture
    def extractor(self):
        return LLMEntityExtractor(
            base_url="https://api.deepseek.com",
            api_key="test-key",
            model="deepseek-v4-pro",
            timeout_seconds=1,
            max_transcript_chars=12000,
        )

    @pytest.fixture
    def patch_create(self, extractor, mocker):
        return mocker.patch.object(
            extractor._client.chat.completions,
            "create",
            new_callable=AsyncMock,
        )

    def _make_response(self, content: str) -> MagicMock:
        response = MagicMock()
        response.choices[0].message.content = content
        return response

    async def test_valid_json_parses_entities(self, extractor, patch_create):
        patch_create.return_value = self._make_response(
            '{"entities": [{"name": "Dune", "type": "book", "context": "Frank Herbert"}]}'
        )
        result = await extractor.extract("I love Dune by Frank Herbert", None)
        assert len(result) == 1
        assert result[0].name == "Dune"
        assert result[0].type == EntityType.BOOK
        assert result[0].context == "Frank Herbert"

    async def test_blank_names_are_dropped(self, extractor, patch_create):
        patch_create.return_value = self._make_response(
            '{"entities": [{"name": "   ", "type": "book"}, {"name": "Dune", "type": "book"}]}'
        )
        result = await extractor.extract("some text", None)
        assert len(result) == 1
        assert result[0].name == "Dune"

    async def test_empty_text_returns_empty_list(self, extractor, patch_create):
        result = await extractor.extract("", None)
        assert result == []
        patch_create.assert_not_called()

    async def test_whitespace_only_returns_empty_list(self, extractor, patch_create):
        result = await extractor.extract("   \t\n ", None)
        assert result == []
        patch_create.assert_not_called()

    async def test_truncates_at_max_chars(self, mocker):
        extractor = LLMEntityExtractor(
            base_url="https://api.deepseek.com",
            api_key="test-key",
            model="deepseek-v4-pro",
            timeout_seconds=1,
            max_transcript_chars=10,
        )
        mock_create = mocker.patch.object(
            extractor._client.chat.completions,
            "create",
            new_callable=AsyncMock,
        )
        mock_create.return_value = self._make_response('{"entities": []}')
        long_text = "This is a very long transcript that should be truncated"
        await extractor.extract(long_text, None)
        user_message = mock_create.call_args.kwargs["messages"][1]["content"]
        assert user_message == long_text[:10]

    async def test_retry_on_invalid_json_succeeds(self, extractor, patch_create):
        bad_response = self._make_response("not valid json")
        good_response = self._make_response('{"entities": [{"name": "Dune", "type": "book"}]}')
        patch_create.side_effect = [bad_response, good_response]
        result = await extractor.extract("some text", "en")
        assert len(result) == 1
        assert result[0].name == "Dune"
        assert patch_create.call_count == 2

    async def test_both_attempts_invalid_json_raises(self, extractor, patch_create):
        bad_response = self._make_response("still not json")
        patch_create.side_effect = [bad_response, bad_response]
        with pytest.raises(EntityExtractionError) as exc_info:
            await extractor.extract("some text", None)
        assert exc_info.value.original_error is not None
        assert patch_create.call_count == 2

    async def test_api_error_on_both_attempts_raises(self, extractor, patch_create):
        api_error = openai.APIError(
            "boom",
            request=httpx.Request("POST", "https://api.deepseek.com"),
            body=None,
        )
        patch_create.side_effect = [api_error, api_error]
        with pytest.raises(EntityExtractionError) as exc_info:
            await extractor.extract("some text", None)
        assert isinstance(exc_info.value.original_error, openai.APIError)
        assert patch_create.call_count == 2
