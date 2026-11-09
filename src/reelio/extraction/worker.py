"""One-slot execution for claimed Source Extraction jobs."""

import asyncio

from reelio.extraction.jobs import ExtractionJobStore
from reelio.extraction.response_mapper import map_extraction_response
from reelio.extraction.service import ExtractionPipelineProtocol


class ExtractionJobWorker:
    """Claim and execute at most one Source Extraction job at a time."""

    def __init__(
        self,
        store: ExtractionJobStore,
        pipeline: ExtractionPipelineProtocol,
    ) -> None:
        """Initialize one worker with application-owned storage and pipeline seams.

        Args:
            store: Durable job store used to claim and commit Source jobs.
            pipeline: Reusable Source Extraction pipeline used outside transactions.
        """
        self._store = store
        self._pipeline = pipeline
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
            job = await self._store.claim_next_source()
            if job is None:
                return False

            pipeline_result = await self._pipeline.run(job.source_url, job.market)
            response = map_extraction_response(pipeline_result)
            await self._store.complete_success(job.job_id, response)
            return True
