"""Bounded execution for claimed Source Extraction jobs."""

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
from math import isfinite

from reelio.extraction.jobs import (
    ExtractionJobStateError,
    ExtractionJobStore,
    RunningSourceJob,
)
from reelio.extraction.response_mapper import map_extraction_response
from reelio.extraction.service import ExtractionPipelineProtocol


class ExtractionJobWorker:
    """Execute Source Extraction jobs with bounded slots and renewable attempts."""

    def __init__(
        self,
        store: ExtractionJobStore,
        pipeline: ExtractionPipelineProtocol,
        *,
        concurrency: int = 2,
        heartbeat_interval_seconds: float = 15.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        """Initialize one worker with bounded pipeline and lease-heartbeat execution.

        Args:
            store: Durable job store used to claim and commit Source jobs.
            pipeline: One reusable Source Extraction pipeline shared by every slot.
            concurrency: Maximum number of Source jobs executed at one time.
            heartbeat_interval_seconds: Delay between live-attempt lease renewals.
            sleep: Awaitable delay seam used by heartbeat execution.

        Raises:
            ValueError: If concurrency or heartbeat_interval_seconds is invalid.
        """
        if concurrency <= 0:
            raise ValueError("concurrency must be positive")
        if not isfinite(heartbeat_interval_seconds) or heartbeat_interval_seconds <= 0:
            raise ValueError("heartbeat_interval_seconds must be finite and positive")

        self._store = store
        self._pipeline = pipeline
        self._concurrency = concurrency
        self._heartbeat_interval_seconds = heartbeat_interval_seconds
        self._sleep = sleep
        self._run_lock = asyncio.Lock()
        self._claim_gate = asyncio.Lock()
        self._shutdown_lock = asyncio.Lock()
        self._shutdown_requested = False
        self._active_attempt_tasks: set[asyncio.Task[object]] = set()

    async def run_once(self) -> bool:
        """Claim, execute, map, and commit one eligible Source Extraction job.

        Returns:
            True only after one mapped successful response is committed, otherwise False
            when no eligible Source job remains or the attempt loses its lease.
        """
        async with self._run_lock:
            return await self._run_next()

    async def run_until_idle(self) -> int:
        """Process eligible Source jobs in bounded shared-pipeline slots until none remain.

        Returns:
            The number of Source jobs whose successful responses were committed.
        """
        async with self._run_lock:
            completed_job_counts = await asyncio.gather(
                *(self._run_slot_until_idle() for _ in range(self._concurrency))
            )
        return sum(completed_job_counts)

    async def shutdown(self) -> None:
        """Stop future claims and gracefully requeue every still-owned attempt."""
        async with self._shutdown_lock:
            async with self._claim_gate:
                self._shutdown_requested = True
                active_attempt_tasks = tuple(self._active_attempt_tasks)

            for attempt_task in active_attempt_tasks:
                attempt_task.cancel()
            if active_attempt_tasks:
                await asyncio.gather(*active_attempt_tasks, return_exceptions=True)

    async def _run_slot_until_idle(self) -> int:
        """Run one worker slot until no eligible job remains."""
        completed_job_count = 0
        while await self._run_next():
            completed_job_count += 1
        return completed_job_count

    async def _run_next(self) -> bool:
        """Claim and execute one tracked Source Extraction attempt."""
        async with self._claim_gate:
            if self._shutdown_requested:
                return False
            job = await self._store.claim_next_source()
            if job is None:
                return False

            attempt_task = asyncio.current_task()
            if attempt_task is None:
                raise RuntimeError("Source Extraction attempts must run in an asyncio task")
            self._active_attempt_tasks.add(attempt_task)

        try:
            return await self._run_claimed_attempt(job)
        except asyncio.CancelledError:
            if self._shutdown_requested:
                await self._store.requeue_source_attempt(job.job_id, job.lease_token)
                return False
            raise
        finally:
            self._active_attempt_tasks.discard(attempt_task)

    async def _run_claimed_attempt(self, job: RunningSourceJob) -> bool:
        """Execute one owned attempt beside its lease heartbeat."""
        pipeline_task = asyncio.create_task(self._pipeline.run(job.source_url, job.market))
        heartbeat_task = asyncio.create_task(self._heartbeat(job, pipeline_task))
        try:
            completed_tasks, _ = await asyncio.wait(
                (pipeline_task, heartbeat_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if heartbeat_task in completed_tasks and not await heartbeat_task:
                pipeline_task.cancel()
                with suppress(asyncio.CancelledError):
                    await pipeline_task
                return False

            pipeline_result = await pipeline_task
        finally:
            heartbeat_task.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat_task
            if not pipeline_task.done():
                pipeline_task.cancel()
                with suppress(asyncio.CancelledError):
                    await pipeline_task

        response = map_extraction_response(pipeline_result)
        try:
            await self._store.complete_success(job.job_id, job.lease_token, response)
        except ExtractionJobStateError:
            return False
        return True

    async def _heartbeat(
        self,
        job: RunningSourceJob,
        pipeline_task: asyncio.Task[object],
    ) -> bool:
        """Renew one attempt until it completes or its durable ownership is lost."""
        while not pipeline_task.done():
            await self._sleep(self._heartbeat_interval_seconds)
            if pipeline_task.done():
                return True
            if not await self._store.renew_source_attempt(job.job_id, job.lease_token):
                return False
        return True
