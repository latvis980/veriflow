# utils/async_utils.py
"""
Async utilities for mixing sync and async code in Flask + gevent

This module provides utilities to properly handle async code in a Flask (WSGI) environment
running with gunicorn gevent workers.

THE CORE PROBLEM THIS SOLVES:
    Gunicorn gevent workers monkey-patch threading.Thread so it creates greenlets,
    not real OS threads. Multiple concurrent greenlets running in the same OS thread
    share the same threading.get_ident() value. This means a per-thread event loop
    dict causes concurrent jobs to collide on each other's loops, producing:
        "Cannot run the event loop while another loop is running"

THE SOLUTION:
    concurrent.futures.ThreadPoolExecutor creates REAL OS threads even under gevent.
    Each real OS thread gets its own isolated asyncio.run() call with its own event loop.
    asyncio.run() is specifically designed for "run a coroutine from scratch in a new loop"
    and handles all setup/teardown safely.

    The calling greenlet blocks on future.result(), which is gevent-safe -- gevent
    yields control to other greenlets while the real OS thread does async work.

Key utilities:
- run_async_in_thread: Run async code in a real OS thread (safe under gevent)
- cleanup_thread_loop: No-op kept for backward compatibility
- sync_to_async: Convert any callable (sync or async) into an awaitable
"""

import asyncio
import inspect
import concurrent.futures
from functools import wraps, partial
from typing import Any, Callable, Coroutine, TypeVar
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')

# Real OS thread pool for running asyncio work.
# max_workers=20 supports up to 20 concurrent analysis jobs.
# Created at import time -- each submitted task gets a real OS thread
# with its own isolated event loop via asyncio.run().
_ASYNC_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=20,
    thread_name_prefix="veriflow_async"
)


def _translate_api_error(e: Exception) -> Exception:
    """
    Translate raw API exceptions into user-readable messages.

    Single place to map provider-specific error codes to friendly strings.
    Called by run_async_in_thread so ALL async operations across ALL modes
    get consistent error translation automatically.

    Add new providers or error codes here as needed.
    """
    error_str = str(e)

    # OpenAI quota exhausted - account has no remaining credits
    if "insufficient_quota" in error_str or (
        hasattr(e, "code") and getattr(e, "code", None) == "insufficient_quota"
    ):
        return Exception(
            "OpenAI API quota exceeded. The service has run out of credits. "
            "Please top up at https://platform.openai.com/account/billing"
        )

    # OpenAI temporary rate limit (different from quota - just needs a wait)
    if "rate_limit_exceeded" in error_str:
        return Exception(
            "OpenAI API rate limit reached. Please wait a moment and try again."
        )

    # Fallback: "'error'" means a quota error was mangled somewhere in the async chain
    if error_str == "'error'":
        return Exception(
            "OpenAI API quota exceeded. The service has run out of credits. "
            "Please top up at https://platform.openai.com/account/billing"
        )

    return e


def run_async_in_thread(coro: Coroutine[Any, Any, T]) -> T:
    """
    Run an async coroutine in a real OS thread with its own event loop.

    This is the ONLY function you need to call to run async code from sync
    Flask routes or background task runners.

    Why real OS threads instead of asyncio loop reuse:
        Under gunicorn + gevent, threading.Thread creates greenlets. Multiple
        greenlets in the same OS thread share the same thread ID, so a
        per-thread loop dict collides between concurrent jobs. Real OS threads
        from ThreadPoolExecutor each have unique IDs and isolated event loops.

    Why asyncio.run() inside the thread:
        asyncio.run() creates a fresh event loop, runs the coroutine to
        completion, then closes the loop. This is the safest and simplest
        pattern -- no manual loop lifecycle management needed.

    Why future.result() is gevent-safe:
        Blocking on future.result() yields control to other gevent greenlets
        while the real OS thread does async work. No deadlock occurs.

    Args:
        coro: Coroutine to run

    Returns:
        Result of the coroutine

    Raises:
        Exception with a user-readable message for known API errors
        (quota exhausted, rate limits). See _translate_api_error.

    Usage:
        result = run_async_in_thread(my_async_function(arg1, arg2))
    """
    future = _ASYNC_EXECUTOR.submit(asyncio.run, coro)
    try:
        return future.result()
    except Exception as e:
        raise _translate_api_error(e) from None


def cleanup_thread_loop():
    """
    No-op kept for backward compatibility.

    Previously managed per-thread event loop lifecycle. No longer needed
    because asyncio.run() handles full loop lifecycle within each real OS thread.

    Safe to call -- does nothing.
    """
    pass


def sync_to_async(func: Callable[..., T]) -> Callable[..., Coroutine[Any, Any, T]]:
    """
    Convert any callable into an awaitable function.

    - If func is already async def -> just await it
    - If func is sync -> run it in ThreadPoolExecutor so event loop never blocks

    Args:
        func: Any callable (sync or async)

    Returns:
        Async wrapper function

    Usage:
        result = await sync_to_async(my_function)(*args, **kwargs)
    """
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        if inspect.iscoroutinefunction(func):
            return await func(*args, **kwargs)

        loop = asyncio.get_running_loop()
        bound = partial(func, *args, **kwargs)
        return await loop.run_in_executor(None, bound)

    return async_wrapper


def safe_float(value, default=0.0):
    """Safely convert a value to float. LLMs sometimes return numbers as strings."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    return default


# ---------------------------------------------------------------------------
# Legacy functions kept for backward compatibility
# ---------------------------------------------------------------------------

def get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    """
    Deprecated. Use run_async_in_thread() instead.
    Kept for any code that imports this directly.
    """
    try:
        loop = asyncio.get_event_loop()
        if not loop.is_closed() and not loop.is_running():
            return loop
    except RuntimeError:
        pass
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop


_PENDING_TASKS: set = set()


def track_async_task(coro: Coroutine) -> Any:
    """Deprecated. Kept for backward compatibility."""
    try:
        task = asyncio.create_task(coro)
        _PENDING_TASKS.add(task)
        task.add_done_callback(_PENDING_TASKS.discard)
        return task
    except RuntimeError:
        return asyncio.run(coro)


async def wait_for_pending_tasks():
    """Wait for all tracked tasks. Kept for backward compatibility."""
    if _PENDING_TASKS:
        logger.info(f"Waiting for {len(_PENDING_TASKS)} pending tasks...")
        try:
            await asyncio.gather(*_PENDING_TASKS, return_exceptions=True)
            logger.info("All pending tasks completed")
        except Exception as e:
            logger.error(f"Error waiting for pending tasks: {e}")


def shutdown_all_loops():
    """
    Shutdown the async executor pool.
    Call during application shutdown for graceful cleanup.
    """
    logger.info("Shutting down async executor pool...")
    _ASYNC_EXECUTOR.shutdown(wait=False)
    logger.info("Async executor pool shut down")