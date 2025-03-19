"""
Enhanced async wrapper utility for Flask routes.
Provides robust error handling and automatic Flask context preservation.
"""

import asyncio
import functools
import logging
import traceback
import sys
from typing import Any, Callable, Coroutine, TypeVar 
from flask import current_app, has_request_context, has_app_context, request, g

T = TypeVar('T')

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

def run_async(coro: Coroutine[Any, Any, T], timeout: float = 30.0) -> T:
    """
    Run an async function in a synchronous context with proper cleanup.
    Safely handles cases where it's called from within an existing event loop.

    Args:
        coro: Coroutine to execute
        timeout: Timeout in seconds

    Returns:
        Result of the coroutine execution
    """
    # Check if we're already in an event loop
    try:
        current_loop = asyncio.get_running_loop()
        # If we get here, we're already in an event loop
        # Use run_coroutine_threadsafe to run in the existing loop from a different thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = asyncio.run_coroutine_threadsafe(coro, current_loop)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                future.cancel()
                logger.warning(f"Async operation timed out after {timeout}s")
                return None
            except Exception as e:
                logger.error(f"Error in threaded async execution: {str(e)}")
                return None
    except RuntimeError:
        # No event loop running, create a new one
        loop = asyncio.new_event_loop()

        try:
            # Set the loop as current
            asyncio.set_event_loop(loop)

            # Create a task and run it with timeout
            task = loop.create_task(coro)
            return loop.run_until_complete(asyncio.wait_for(task, timeout=timeout))
        finally:
            # Clean up: cancel all pending tasks
            pending_tasks = [task for task in asyncio.all_tasks(loop)
                            if not task.done() and task != asyncio.current_task(loop)]

            if pending_tasks:
                for task in pending_tasks:
                    task.cancel()

                # Wait for tasks to cancel
                if loop.is_running():
                    loop.create_task(asyncio.gather(*pending_tasks, return_exceptions=True))
                else:
                    loop.run_until_complete(asyncio.gather(*pending_tasks, return_exceptions=True))

            # Close the loop
            loop.close()

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
