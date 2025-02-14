import logging
import os
import datetime
from config import LOG_DIR, LOG_LEVEL


class LoggerManager:
    _instance = None

    def __new__(cls, log_dir=LOG_DIR):
        if cls._instance is None:
            cls._instance = super(LoggerManager, cls).__new__(cls)
            cls._instance._initialize(log_dir)
        return cls._instance

    def _initialize(self, log_dir):
        """Initialize logging configuration."""
        os.makedirs(log_dir, exist_ok=True) 

        log_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_file = os.path.join(log_dir, f"log_{log_timestamp}.log")

        logging.basicConfig(
            level=LOG_LEVEL,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(self.log_file, mode="a"),
                # logging.StreamHandler() # Logging to console
            ],
        )

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging initialized. Logs will be saved to {self.log_file}")

        # Suppress noisy logs from specific libraries
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)

    def get_logger(self):
        return self.logger
