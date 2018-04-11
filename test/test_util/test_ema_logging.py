'''
Created on Jul 28, 2015

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
'''
from __future__ import (absolute_import, print_function, division,
                        unicode_literals)
import logging

try:
    import unittest.mock as mock
except ImportError:
    import mock
import unittest

from ema_workbench.util import ema_logging

def tearDownModule():
    ema_logging._logger = None
    ema_logger = logging.getLogger(ema_logging.LOGGER_NAME)
    ema_logger.handlers = []

class TestEmaLogging(unittest.TestCase):

    def test_log_messages(self):
        ema_logging.log_to_stderr(ema_logging.DEBUG)
        
        with mock.patch('ema_workbench.util.ema_logging._logger') as mocked_logger:
            message = 'test message'
            ema_logging.debug(message)
            mocked_logger.debug.assert_called_with(message)

            ema_logging.info(message)
            mocked_logger.info.assert_called_with(message)
            
            ema_logging.warning(message)
            mocked_logger.warning.assert_called_with(message)
            
            ema_logging.error(message)
            mocked_logger.error.assert_called_with(message)
            
            ema_logging.exception(message)
            mocked_logger.exception.assert_called_with(message)
            
            ema_logging.critical(message)
            mocked_logger.critical.assert_called_with(message)            

    def test_get_logger(self):
        ema_logging._logger = None
        logger = ema_logging.get_logger()
        self.assertEqual(logger, logging.getLogger(ema_logging.LOGGER_NAME))
        self.assertEqual(len(logger.handlers), 1)
        self.assertEqual(type(logger.handlers[0]), ema_logging.NullHandler)
        
        logger = ema_logging.get_logger()
        self.assertEqual(logger, logging.getLogger(ema_logging.LOGGER_NAME))
        self.assertEqual(len(logger.handlers), 1)
        self.assertEqual(type(logger.handlers[0]), ema_logging.NullHandler)        
    
    def test_log_to_stderr(self):
        ema_logging._logger = None
        logger = ema_logging.log_to_stderr(ema_logging.DEBUG)
        self.assertEqual(len(logger.handlers), 2)
        self.assertEqual(logger.level, ema_logging.DEBUG)
        

        ema_logging._logger = None
        logger = ema_logging.log_to_stderr()
        self.assertEqual(len(logger.handlers), 2)
        self.assertEqual(logger.level, ema_logging.DEFAULT_LEVEL)
        
        logger = ema_logging.log_to_stderr()
        self.assertEqual(len(logger.handlers), 2)
        self.assertEqual(logger.level, ema_logging.DEFAULT_LEVEL)


if __name__ == "__main__":
    unittest.main()