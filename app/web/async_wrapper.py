"""
Enhanced async wrapper utility for Flask routes.
Provides robust error handling and automatic Flask context preservation.
"""

import asyncio
import functools
import logging
import traceback
import sys
from typing import Any, Callable, Coroutine
from flask import current_app, has_request_context, has_app_context, request, g

logger = logging.getLogger(__name__)

def async_handler(func):
    """
    Decorator to handle async functions in Flask routes, properly preserving Flask context.

    Args:
        func: An async function to be used in a Flask route

    Returns:
        A synchronous function that can be used in Flask routes
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Check if we're in a request context
        if has_request_context():
            # This is a route function, we need to preserve request context
            # Instead of directly accessing _request_ctx_stack, capture current request and context
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
        else:
            # Not in a request context, just run normally
            coroutine = func(*args, **kwargs)
            if asyncio.iscoroutine(coroutine):
                return run_async(coroutine)
            else:
                return coroutine

    return wrapper

def run_async(coro: Coroutine) -> Any:
    """
    Run an async function in a synchronous context with enhanced error handling.
    """
    try:
        # Use a fresh event loop for each run to avoid contamination
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Configure the loop with a larger thread pool
        import concurrent.futures
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
        loop.set_default_executor(executor)

        try:
            # Preserve app context if needed
            if has_app_context():
                app_context = current_app._get_current_object().app_context()

                # Define function to run with app context
                async def run_with_app_context():
                    with app_context:
                        return await coro

                # Run with app context preservation
                result = loop.run_until_complete(run_with_app_context())
            else:
                # No app context to preserve
                result = loop.run_until_complete(coro)

            return result
        finally:
            # Properly clean up all resources
            try:
                # Cancel pending tasks
                tasks = [t for t in asyncio.all_tasks(loop) if not t.done()]
                if tasks:
                    for task in tasks:
                        task.cancel()
                    loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
            except Exception as cleanup_err:
                logger.error(f"Error during task cleanup: {str(cleanup_err)}")
            finally:
                # Don't immediately close - run loop a bit longer to process cancellations
                loop.run_until_complete(asyncio.sleep(0.1))
                loop.close()
                executor.shutdown(wait=False)
    except Exception as e:
        logger.error(f"Event loop error: {str(e)}")
        logger.error(traceback.format_exc())
        raise

# Add specialized task management utilities
def create_background_task(coro: Coroutine) -> asyncio.Task:
    """
    Create a background task that will run independently.
    Useful for fire-and-forget operations that shouldn't block responses.

    Args:
        coro: Coroutine to execute in background

    Returns:
        Task object
    """
    try:
        loop = asyncio.get_event_loop()
        if not loop.is_running():
            # Can't create background task in non-running loop
            raise RuntimeError("Event loop is not running")

        # Create background task
        task = loop.create_task(coro)

        # Add done callback to log any errors
        def on_task_done(task):
            try:
                # Try to retrieve result to show errors
                task.result()
            except asyncio.CancelledError:
                logger.debug("Background task was cancelled")
            except Exception as e:
                logger.error(f"Error in background task: {str(e)}")
                logger.error(traceback.format_exc())

        task.add_done_callback(on_task_done)

        logger.debug(f"Created background task: {task}")
        return task
    except Exception as e:
        logger.error(f"Failed to create background task: {str(e)}")
        logger.error(traceback.format_exc())
        raise
