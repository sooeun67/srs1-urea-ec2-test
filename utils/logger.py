import logging
import time
from datetime import datetime
from logging import LogRecord
from typing import Optional

from pytz import timezone


class TzFormatter(logging.Formatter):
    def converter(self, timestamp):
        """timestamp converter to given timezone

        Parameters
        ----------
        timestamp : _type_

        Returns
        -------
        converted time
        """
        if self.timezone:
            return datetime.fromtimestamp(timestamp, tz=timezone(self.timezone))
        else:
            return time.localtime(timestamp)

    def formatTime(self, record: LogRecord, datefmt: Optional[str] = None) -> str:
        """format time of given record

        Parameters
        ----------
        record : LogRecord
        datefmt : Optional[str], optional, by default None

        Returns
        -------
        str
        """
        if self.timezone:
            ct = self.converter(record.created)
            if datefmt:
                s = ct.strftime(datefmt)
            else:
                s = ct.strftime(self.default_time_format)
                if self.default_msec_format:
                    s = self.default_msec_format % (s, record.msecs)
            return s
        else:
            return super().formatTime(record, datefmt)

    def set_timezone(self, timezone_name):
        """set timezone

        Parameters
        ----------
        timezone_name
        """
        self.timezone = timezone_name


def set_logger(
    logger_name: str,
    logger_format: str = "%(asctime)s [%(levelname)s]: %(message)s",
    level: int = logging.INFO,
    timezone_name: str = "UTC",
) -> None:
    """set logger's timezone and format

    Parameters
    ----------
    logger_name : str
    logger_format : _type_, optional, by default "%(asctime)s [%(levelname)s]: %(message)s"
    level : int, optional, by default logging.INFO
    timezone_name : str, optional, by default "UTC"
    """

    logger = logging.getLogger(logger_name)
    formatter = TzFormatter(logger_format, datefmt="%Y/%m/%d %H:%M:%S %Z")
    formatter.set_timezone(timezone_name)
    logger.setLevel(level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.propagate = False  # Prevent duplicate logs in AWS Lambda
