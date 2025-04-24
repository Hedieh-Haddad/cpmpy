import sys

try:
    from loguru import logger
    use_loguru = True
except ImportError:
    use_loguru = False

def log(msg, level="info"):
    """
    Print a log message using loguru if available, or fallback to stderr.

    Args:
        msg (str): The message to log.
        level (str): One of 'info', 'warning', 'error', 'debug'.
    """
    if use_loguru:
        log_fn = getattr(logger, level, logger.info)
        log_fn(msg)
    else:
        prefix = {
            "info": "[INFO]",
            "warning": "[WARNING]",
            "error": "[ERROR]",
            "debug": "[DEBUG]"
        }.get(level, "[INFO]")
        print(f"{prefix} {msg}", file=sys.stderr)