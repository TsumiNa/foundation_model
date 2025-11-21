import sys

import torch
from loguru import logger

# Set float32 matmul precision globally for the package
# See https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
torch.set_float32_matmul_precision("high")

# Configure logger for this package
# Remove default handler to avoid duplicate logs
logger.remove()

# Add a new handler with a specific format and level
# This format includes timestamp, level, module, function, line number, and message
# Filter to only show logs from foundation_model package to reduce noise from worker processes
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",  # Set default logging level to INFO
    colorize=True,
)

# Example file handler (optional, uncomment and customize if needed):
# logger.add(
#     "logs/foundation_model_{time}.log",
#     rotation="10 MB",  # Rotate log file when it reaches 10 MB
#     retention="10 days", # Keep logs for 10 days
#     enqueue=True,      # Asynchronous logging
#     format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
#     level="DEBUG"        # Log DEBUG level and above to file
# )
