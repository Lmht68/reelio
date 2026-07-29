import json
import logging

import openai
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, ValidationError

from src.entities.exceptions import EntityExtractionError
from src.entities.extractors.base import EntityExtractor
from src.entities.models import Entity

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You extract entertainment entities from short-video transcripts. "
    "Only extract entities explicitly mentioned in the transcript. "
    "Do not infer or hallucinate titles, artists, directors, authors, or other information. "
    "Extract movies, directors, songs, albums, artists, books, and authors that are recommended, "
    "praised, discussed positively, or otherwise highlighted by the speaker. "
    "Normalize each entity to its widely recognized canonical name. "
    "For works, use the official original release title when possible. "
    "For people, use their commonly known professional name. "
    "Deduplicate identical entities. "
    'Respond only with valid JSON of the form '
    '{"entities":[{"name":"...","type":"...","context":"..."}]}. '
    '"type" must be one of: movie, director, song, album, artist, book, author. '
    'Include "context" only when the transcript explicitly provides information needed '
    'to disambiguate the entity (for example, artist, author, or release year). '
    'Never invent context. '
    'If no qualifying entities are found, return {"entities":[]}.'
)


class _LLMEntityList(BaseModel):
    entities: list[Entity]


class LLMEntityExtractor(EntityExtractor):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float,
        max_transcript_chars: int,
    ):
        self._client = openai.AsyncOpenAI(
            base_url=base_url, api_key=api_key, timeout=timeout_seconds
        )
        self._model = model
        self._max_transcript_chars = max_transcript_chars

    async def extract(self, text: str, language: str | None) -> list[Entity]:
        if not text.strip():
            return []

        text = text[: self._max_transcript_chars]

        system_content = _SYSTEM_PROMPT
        if language is not None:
            system_content += (
                f" The transcript language is '{language}'; "
                "keep canonical names in their original language."
            )
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": text},
        ]

        for attempt in range(2):
            try:
                response = await self._client.chat.completions.create(
                    model=self._model,
                    temperature=0,
                    response_format={"type": "json_object"},
                    messages=messages,
                )
                raw = response.choices[0].message.content
                assert raw is not None  # json_object response always has content
                data = json.loads(raw)
                parsed = _LLMEntityList.model_validate(data)
            except (openai.APIError, json.JSONDecodeError, ValidationError) as exc:
                if attempt == 1:
                    logger.warning(
                        "Entity extraction failed after retry: %s", exc
                    )
                    raise EntityExtractionError(
                        "Failed to extract entities after retry", original_error=exc
                    ) from exc
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your previous response could not be processed ({exc}). "
                            "Respond again with valid JSON only."
                        ),
                    }
                )
                continue
            else:
                break

        return [entity for entity in parsed.entities if entity.name.strip()]
