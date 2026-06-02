# -*- coding: utf-8 -*-
"""
Utility: safe execution of CPU‑bound inference with configurable timeout.
If the callable exceeds `timeout_sec`, a sentinel dict with
`available=False` and `reason="timeout"` is returned.
"""

import concurrent.futures
import logging
import time
from typing import Any, Callable, Dict

logger = logging.getLogger("smartzi.timeout")


def run_with_timeout(
    func: Callable[..., Dict[str, Any]],
    *args,
    timeout_sec: int = 60,
    **kwargs,
) -> Dict[str, Any]:
    """Execute `func` in an isolated thread with a hard timeout.
    Returns the function's dict on success, otherwise a timeout sentinel.
    """
    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            result = future.result(timeout=timeout_sec)
            # Attach processing time for downstream logging
            result["processing_time"] = round(time.monotonic() - start, 3)
            return result
        except concurrent.futures.TimeoutError:
            logger.error(
                "Inference timeout after %s s in %s",
                timeout_sec,
                func.__qualname__,
            )
            return {"available": False, "reason": "timeout"}
        except Exception as exc:  # pragma: no cover – defensive
            logger.exception("Unexpected error in %s", func.__qualname__)
            return {"available": False, "reason": "inference_failed"}
