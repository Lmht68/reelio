"""Bounded execution for claimed Source Extraction jobs."""

import asyncio

from reelio.extraction.jobs import ExtractionJobStore
from reelio.extraction.response_mapper import map_extraction_response
from reelio.extraction.service import ExtractionPipelineProtocol


class ExtractionJobWorker:
    """Execute Source Extraction jobs with a bounded number of shared-pipeline slots."""

    def __init__(
        self,
        store: ExtractionJobStore,
        pipeline: ExtractionPipelineProtocol,
        *,
        concurrency: int = 2,
    ) -> None:
        """Initialize one worker with application-owned storage and pipeline seams.

        Args:
            store: Durable job store used to claim and commit Source jobs.
            pipeline: One reusable Source Extraction pipeline shared by every slot.
            concurrency: Maximum number of Source jobs executed at one time.

        Raises:
            ValueError: If concurrency is not positive.
        """
        if concurrency <= 0:
            raise ValueError("concurrency must be positive")

        self._store = store
        self._pipeline = pipeline
        self._concurrency = concurrency
        self._run_lock = asyncio.Lock()

    async def run_once(self) -> bool:
        """Claim, execute, map, and commit one queued Source Extraction job.

        Returns:
            True only after one mapped successful response is committed, otherwise False
            when no queued Source job is available.

        Raises:
            ExtractionJobStoreUnavailableError: If durable storage cannot claim or
                complete the job.
            ExtractionJobStateError: If the claimed job cannot complete.
            ExtractionError: If Source pipeline execution fails.
        """
        async with self._run_lock:
            return await self._run_next()

    async def run_until_idle(self) -> int:
        """Process queued Source jobs in bounded shared-pipeline slots until none remain.

        Returns:
            The number of Source jobs whose successful responses were committed.

        Raises:
            ExtractionJobStoreUnavailableError: If durable storage cannot claim or
                complete a job.
            ExtractionJobStateError: If a claimed job cannot complete.
            ExtractionError: If Source pipeline execution fails.
        """
        async with self._run_lock:
            completed_job_counts = await asyncio.gather(
                *(self._run_slot_until_idle() for _ in range(self._concurrency))
            )
        return sum(completed_job_counts)

    async def _run_slot_until_idle(self) -> int:
        """Run one worker slot until no queued job remains."""
        completed_job_count = 0
        while await self._run_next():
            completed_job_count += 1
        return completed_job_count

    async def _run_next(self) -> bool:
        """Claim, execute, map, and commit one queued Source Extraction job."""
        job = await self._store.claim_next_source()
        if job is None:
            return False

        pipeline_result = await self._pipeline.run(job.source_url, job.market)
        response = map_extraction_response(pipeline_result)
        await self._store.complete_success(job.job_id, response)
        return True
