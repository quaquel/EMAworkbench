'''
Created on Jul 28, 2015

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
'''
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


if __name__ == "__main__":
    unittest.main()