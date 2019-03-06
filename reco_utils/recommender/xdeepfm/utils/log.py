"""define logging configure"""
import logging
from datetime import datetime, timedelta, timezone
import platform

__all__ = ["Log"]
class Log(object):
    def __init__(self, hparams):
        # UTC To Beijing Time
        utc_dt = datetime.utcnow().replace(tzinfo=timezone.utc)
        bj_dt = utc_dt.astimezone(timezone(timedelta(hours=8)))

        logging_filename = "logs/"+hparams.log + '__' + bj_dt.strftime('%Y-%m-%d_%H_%M_%S') + '.log'
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(logging_filename)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
