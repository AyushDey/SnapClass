import logging
import sys
from logging.handlers import RotatingFileHandler

def setup_logger(name: str = "snapclass", log_file: str = "app.log", level=logging.INFO):
    """
    Sets up a logger with both console and file output.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # File Handler (Rotating)
    file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    file_handler.setFormatter(formatter)

    # Avoid adding handlers multiple times if setup_logger is called repeatedly
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger

def intercept_uvicorn_logs():
    """
    Replaces Uvicorn's default loggers with our custom configuration
    so that all logs (server & app) look the same.
    """
    # Names of uvicorn loggers
    loggers = ["uvicorn", "uvicorn.access", "uvicorn.error"]
    
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    for name in loggers:
        logger = logging.getLogger(name)
        # Remove default handlers
        logger.handlers = []
        logger.propagate = False # Prevent double logging if root logger captures it
        
        # Add our handlers (Console + File)
        # Note: In a real heavy-load prod, you might separate access logs, 
        # but for this scale, unified is better for debugging.
        
        # Console
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)
        logger.addHandler(console)
        
        # File (re-use the same file for all logs)
        file_h = RotatingFileHandler("app.log", maxBytes=10*1024*1024, backupCount=5)
        file_h.setFormatter(formatter)
        logger.addHandler(file_h)
