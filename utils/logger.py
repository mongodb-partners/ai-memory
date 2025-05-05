import sys
import config
import asyncio
import logging
import os
import time
import inspect
import functools
from contextlib import contextmanager
from httpx import AsyncClient, Client, HTTPStatusError
from datetime import datetime, timezone

def get_logger():
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            stream=sys.stdout
        )
        return logging.getLogger(config.APP_NAME)

# Export logger
logger = get_logger()