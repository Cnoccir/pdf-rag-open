# In app/web/async_wrapper.py

"""
Enhanced async wrapper utility for Flask routes and Celery tasks.
Provides robust error handling and automatic Flask/Celery context preservation.
"""

import asyncio
import functools
import logging
import traceback
import sys
from typing import Any, Callable, Coroutine, TypeVar, Optional
from flask import current_app, has_request_context, has_app_context, request, g

T = TypeVar('T')

logger = logging.getLogger(__name__)

def async_handler(func):
    """
    Decorator to handle async functions in Flask routes, properly preserving Flask context.
    Also works with Celery tasks by detecting execution environment.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Detect if we're in a Celery worker
        is_celery = 'celery' in sys.modules and hasattr(sys.modules['celery'], 'current_task') and sys.modules['celery'].current_task

        # Check if we're in a request context (Flask routes)
        if has_request_context():
            # This is a route function, we need to preserve request context
            current_request = request
            app_context = current_app.app_context()

            # Capture g variables that might be needed
            g_vars = {}
            for key in dir(g):
                if not key.startswith('_'):
                    try:
                        g_vars[key] = getattr(g, key)
                    except AttributeError:
                        pass

            # Define a function that preserves request context
            async def run_with_context():
                with app_context:
                    # Use the captured request context
                    with current_app.test_request_context(
                        environ_base=current_request.environ
                    ):
                        # Restore g variables
                        for key, value in g_vars.items():
                            setattr(g, key, value)
                        return await func(*args, **kwargs)

            # Run the function with context preservation
            return run_async(run_with_context())

        # If in Celery task or no request context
        elif is_celery or has_app_context():
            # For Celery tasks, we need to handle app context properly
            app_context = current_app.app_context() if has_app_context() else None

            async def run_with_celery_context():
                if app_context:
                    with app_context:
                        return await func(*args, **kwargs)
                else:
                    return await func(*args, **kwargs)

            # Use a special version that's more Celery-friendly
            return run_async_celery(run_with_celery_context())
        else:
            # Not in a request context, just run normally
            coroutine = func(*args, **kwargs)
            if asyncio.iscoroutine(coroutine):
                return run_async(coroutine)
            else:
                return coroutine

    return wrapper

def run_async(coro: Coroutine[Any, Any, T], timeout: float = 30.0) -> T:
    """
    Run an async function in a synchronous context with proper cleanup.
    Safely handles cases where it's called from within an existing event loop.
    """
    # Use a safer approach for detecting existing loops to prevent errors
    try:
        # Check if there's already a running loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're in a running loop, use run_coroutine_threadsafe
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = asyncio.run_coroutine_threadsafe(coro, loop)
                try:
                    return future.result(timeout=timeout)
                except concurrent.futures.TimeoutError:
                    future.cancel()
                    logger.warning(f"Async operation timed out after {timeout}s")
                    return None
        else:
            # There's a loop but it's not running
            return loop.run_until_complete(asyncio.wait_for(coro, timeout=timeout))

    except RuntimeError:
        # No event loop running, create a new one
        return run_with_new_loop(coro, timeout)

def run_async_celery(coro: Coroutine[Any, Any, T], timeout: float = 300.0) -> T:
    """
    Special version for Celery tasks with longer timeout and better error handling.
    """
    try:
        return run_with_new_loop(coro, timeout)
    except Exception as e:
        logger.error(f"Error in Celery async execution: {str(e)}", exc_info=True)
        # Let Celery handle the error for retry logic
        raise

def run_with_new_loop(coro: Coroutine[Any, Any, T], timeout: float) -> T:
    """
    Run a coroutine in a new event loop with proper cleanup.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        # Create a task and run it with timeout
        return loop.run_until_complete(asyncio.wait_for(coro, timeout=timeout))
    finally:
        # Clean up: cancel all pending tasks
        try:
            # Get all tasks, being careful with asyncio API differences
            if hasattr(asyncio, 'all_tasks'):
                pending_tasks = asyncio.all_tasks(loop)
            else:
                pending_tasks = asyncio.Task.all_tasks(loop)

            # Filter current task
            current_task = asyncio.current_task(loop) if hasattr(asyncio, 'current_task') else None
            pending_tasks = [t for t in pending_tasks if t != current_task]

            if pending_tasks:
                # Cancel all tasks
                for task in pending_tasks:
                    task.cancel()

                # Wait for tasks to cancel
                if loop.is_running():
                    loop.create_task(asyncio.gather(*pending_tasks, return_exceptions=True))
                else:
                    loop.run_until_complete(asyncio.gather(*pending_tasks, return_exceptions=True))
        except Exception as cleanup_error:
            logger.warning(f"Error during event loop cleanup: {str(cleanup_error)}")

        finally:
            # Close the loop
            loop.close()
