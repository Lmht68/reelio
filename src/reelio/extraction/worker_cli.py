"""Long-running process entry point for durable Extraction Job execution."""

import asyncio
import logging
import signal
from contextlib import AsyncExitStack, suppress

from reelio.config import app_settings
from reelio.extraction.factory import create_production_extraction_pipeline
from reelio.extraction.job_config import ExtractionJobConfig
from reelio.extraction.job_store import create_production_extraction_job_store
from reelio.extraction.worker import ExtractionJobWorker
from reelio.logging import configure_logging

logger = logging.getLogger(__name__)


async def run_worker(stop_event: asyncio.Event | None = None) -> None:
    """Own production worker resources and process durable jobs until stopped."""
    settings = ExtractionJobConfig()
    requested_stop = stop_event or asyncio.Event()
    event_loop = asyncio.get_running_loop()
    installed_signals: list[signal.Signals] = []

    if stop_event is None:
        for shutdown_signal in (signal.SIGINT, signal.SIGTERM):
            event_loop.add_signal_handler(shutdown_signal, requested_stop.set)
            installed_signals.append(shutdown_signal)

    try:
        async with AsyncExitStack() as exit_stack:
            store = await create_production_extraction_job_store()
            exit_stack.push_async_callback(store.aclose)
            pipeline = await create_production_extraction_pipeline()
            exit_stack.push_async_callback(pipeline.aclose)
            worker = ExtractionJobWorker(
                store,
                pipeline,
                concurrency=settings.worker_concurrency,
                heartbeat_interval_seconds=settings.heartbeat_interval_seconds,
            )
            worker_task = asyncio.create_task(
                worker.run_forever(settings.worker_poll_interval_seconds)
            )
            stop_task = asyncio.create_task(requested_stop.wait())
            logger.info("Extraction Job worker started", extra={"stage": "job_worker"})
            try:
                completed, _ = await asyncio.wait(
                    (worker_task, stop_task),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if worker_task in completed:
                    await worker_task
            finally:
                await worker.shutdown()
                for task in (worker_task, stop_task):
                    if not task.done():
                        task.cancel()
                with suppress(asyncio.CancelledError):
                    await worker_task
                with suppress(asyncio.CancelledError):
                    await stop_task
                logger.info("Extraction Job worker stopped", extra={"stage": "job_worker"})
    finally:
        for shutdown_signal in installed_signals:
            event_loop.remove_signal_handler(shutdown_signal)


def main() -> int:
    """Run the production worker process and return its shell exit status."""
    configure_logging(app_settings.log_level)
    try:
        asyncio.run(run_worker())
    except KeyboardInterrupt:
        return 0
    except Exception:
        logger.error(
            "Extraction Job worker stopped after an unexpected failure",
            extra={"stage": "job_worker", "reason": "unexpected_failure"},
        )
        return 1
    return 0
