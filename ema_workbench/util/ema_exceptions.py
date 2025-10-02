"""Exceptions and warning used internally by the EMA workbench.

In line with advice given in `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_.
"""

# Created on 31 mei 2011
#
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = ["EMAError", "EMAParallelError", "EMAWarning", "ExperimentError"]


class EMAError(BaseException):
    """Base EMA error"""

    def __init__(self, *args):
        self.args = args

    def __str__(self):
        if len(self.args) == 1:
            return str(self.args[0])
        else:
            return str(self.args)

    def __repr__(self):
        return f"{self.__class__.__name__}(*{self.args!r})"


class EMAWarning(EMAError):
    """base EMA warning class"""


class ExperimentError(EMAError):
    """error to be used when a particular run creates an error.

    The character of the error can be specified as the message, and the actual experiment that
    gave rise to the error.

    """

    def __init__(self, message, case, policy=None):
        self.message = message
        self.case = case
        self.args = (message, case)
        try:
            self.policy = policy.name
        except AttributeError:
            self.policy = "None"

    def __str__(self):
        keys = sorted(self.case.keys())

        c = ""
        for key in keys:
            value = self.case.get(key)
            c += key
            c += ":"
            c += str(value)
            c += ", "
        c += "policy:" + self.policy

        return self.message + " case: {" + c + "}"

    def __repr__(self):
        return f"{self.message} case: {self.case!r} "


class EMAParallelError(EMAError):
    """parallel EMA error"""
