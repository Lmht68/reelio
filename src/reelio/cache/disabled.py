"""No-op cache implementation for explicitly disabled shared caching."""

from collections.abc import Awaitable, Callable

from reelio.cache.interface import (
    CacheEntry,
    CacheSkip,
    CacheWrite,
    RetainedCacheValue,
    RevalidatingCacheEntry,
)


class DisabledCache:
    """Invoke loaders without retaining cached values or opening external resources."""

    async def get_or_load[ValueT](
        self,
        entry: CacheEntry[ValueT],
        loader: Callable[[], Awaitable[ValueT]],
    ) -> ValueT:
        """Load a value exactly once without inspecting cache metadata.

        Args:
            entry: Ignored cache operation descriptor.
            loader: Awaitable operation producing the required value.

        Returns:
            Value returned by the loader.
        """
        del entry
        return await loader()

    async def get_or_load_revalidating[ValueT](
        self,
        entry: RevalidatingCacheEntry[ValueT],
        loader: Callable[
            [RetainedCacheValue[ValueT] | None],
            Awaitable[CacheWrite[ValueT] | CacheSkip[ValueT]],
        ],
    ) -> ValueT:
        """Load once without retaining a value for a later revalidation.

        Args:
            entry: Ignored cache operation descriptor.
            loader: Operation receiving no retained value.

        Returns:
            The loaded normalized value.
        """
        del entry
        return (await loader(None)).value

    async def aclose(self) -> None:
        """Release no resources because disabled caching owns none."""
