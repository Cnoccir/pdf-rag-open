"""
Simplified async wrapper utility for Flask routes with better error handling.
"""

import asyncio
import functools
import logging
import traceback
import sys
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)

def async_handler(func):
    """
    Decorator to handle async functions in Flask routes.

    Args:
        func: An async function to be used in a Flask route

    Returns:
        A synchronous function that can be used in Flask routes
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            coroutine = func(*args, **kwargs)
            if asyncio.iscoroutine(coroutine):
                return run_async(coroutine)
            else:
                return coroutine
        except Exception as e:
            logger.error(f"Error in async handler: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    return wrapper

def run_async(coro: Coroutine) -> Any:
    """
    Run an async function in a synchronous context.

    Args:
        coro: Coroutine to execute

    Returns:
        Result of the coroutine execution
    """
    try:
        # Create a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            return loop.run_until_complete(coro)
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            logger.error(f"Error running async function: {str(e)}")
            logger.error("Exception traceback:")
            traceback_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            for line in traceback_lines:
                logger.error(line.rstrip())
            raise
        finally:
            # Clean up the loop
            try:
                # Cancel any pending tasks
                tasks = [t for t in asyncio.all_tasks(loop) if not t.done()]
                if tasks:
                    for task in tasks:
                        task.cancel()
                    loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
            except Exception as e:
                logger.error(f"Error cleaning up async tasks: {str(e)}")
            finally:
                loop.close()
    except Exception as e:
        logger.error(f"Event loop error: {str(e)}")
        logger.error(traceback.format_exc())
        raise
