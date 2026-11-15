"""Created on Jul 28, 2015

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
"""

import logging
import unittest

from ema_workbench.util import ema_logging


def tearDownModule():
    ema_logging._logger = None
    ema_logger = logging.getLogger(ema_logging.LOGGER_NAME)
    ema_logger.handlers = []


class TestEmaLogging(unittest.TestCase):
    def test_get_logger(self):
        ema_logging._rootlogger = None
        logger = ema_logging.get_rootlogger()
        self.assertEqual(logger, logging.getLogger(ema_logging.LOGGER_NAME))
        self.assertEqual(len(logger.handlers), 1)
        self.assertEqual(type(logger.handlers[0]), logging.NullHandler)

        logger = ema_logging.get_rootlogger()
        self.assertEqual(logger, logging.getLogger(ema_logging.LOGGER_NAME))
        self.assertEqual(len(logger.handlers), 1)
        self.assertEqual(type(logger.handlers[0]), logging.NullHandler)

    def test_log_to_stderr(self):
        ema_logging._rootlogger = None
        logger = ema_logging.log_to_stderr(ema_logging.DEBUG)
        self.assertEqual(len(logger.handlers), 2)
        self.assertEqual(logger.level, ema_logging.DEBUG)

        ema_logging._rootlogger = None
        logger = ema_logging.log_to_stderr()
        self.assertEqual(len(logger.handlers), 2)
        self.assertEqual(logger.level, ema_logging.DEFAULT_LEVEL)

        logger = ema_logging.log_to_stderr()
        self.assertEqual(len(logger.handlers), 2)
        self.assertEqual(logger.level, ema_logging.DEFAULT_LEVEL)


def test_temporary_filter_removed_after_exception():
    """Regression test: the filter is also removed if an exception is raised.

    Previously, the filter stayed in place, so messages kept being filtered
    for the remainder of the session.
    """
    name = "ema_workbench.test_temporary_filter"
    logger = ema_logging.get_module_logger(name)
    assert logger.filters == []

    with ema_logging.temporary_filter(name=name, level=logging.INFO):
        assert len(logger.filters) == 1
    assert logger.filters == []

    try:
        with ema_logging.temporary_filter(name=name, level=logging.INFO):
            assert len(logger.filters) == 1
            raise RuntimeError("some error")
    except RuntimeError:
        pass
    assert logger.filters == []


if __name__ == "__main__":
    unittest.main()
