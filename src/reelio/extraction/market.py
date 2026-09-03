"""Validated effective-market value object for music catalog resolution."""

import re
from typing import Self

from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema, core_schema

_MARKET_PATTERN = re.compile(r"^[A-Z]{2}$")


class SpotifyMarket(str):
    """Represent one validated uppercase ISO 3166-1 alpha-2 Spotify market.

    The value object validates direct construction and participates in Pydantic request,
    response, and environment-settings validation without duplicating the market rule.
    """

    def __new__(cls, value: str) -> Self:
        """Construct a validated Spotify market.

        Args:
            value: Uppercase ISO 3166-1 alpha-2 market code.

        Raises:
            ValueError: If the code does not use uppercase alpha-two syntax.
        """
        if not _MARKET_PATTERN.fullmatch(value):
            raise ValueError("must use uppercase ISO 3166-1 alpha-2 syntax")
        return super().__new__(cls, value)

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: object,
        _handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        """Provide Pydantic validation and JSON serialization for API contracts."""
        return core_schema.no_info_after_validator_function(
            cls,
            core_schema.str_schema(pattern=_MARKET_PATTERN.pattern),
            serialization=core_schema.to_string_ser_schema(),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        schema: CoreSchema,
        handler: GetJsonSchemaHandler,
    ) -> JsonSchemaValue:
        """Expose the market invariant in generated OpenAPI schemas."""
        json_schema = handler(schema)
        json_schema["pattern"] = _MARKET_PATTERN.pattern
        return json_schema
