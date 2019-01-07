
from __future__ import (absolute_import, print_function, unicode_literals,
                        division)

import sys
import os
import glob
from contextlib import contextmanager
import traceback

set_backend = "import matplotlib\nmatplotlib.use('Agg')\n"


@contextmanager
def redirected_output(new_stdout=None, new_stderr=None):
    save_stdout = sys.stdout
    save_stderr = sys.stderr
    if new_stdout is not None:
        sys.stdout = new_stdout
    if new_stderr is not None:
        sys.stderr = new_stderr
    try:
        yield None
    finally:
        sys.stdout = save_stdout
        sys.stderr = save_stderr


def run_exectests(test_dir, log_path='exectests.log'):

    test_files = glob.glob(os.path.join(test_dir, '*.py'))
    test_files = sorted([ex for ex in test_files if ex[0] is not '_'])
    passed = []
    failed = []
    with open(log_path, 'w') as f:
        with redirected_output(new_stdout=f, new_stderr=f):
            for fname in test_files:
                print(">> Executing '%s'" % fname)
                try:
                    code = compile(set_backend + open(fname, 'r').read(),
                                   fname, 'exec')
                    exec(code, {'__name__': '__main__'}, {})
                    passed.append(fname)
                except BaseException:
                    traceback.print_exc()
                    failed.append(fname)
                    pass

    print(">> Passed %i/%i tests: " % (len(passed), len(test_files)))
    print("Passed: " + ', '.join(passed))
    print("Failed: " + ', '.join(failed))
    print("See %s for details" % log_path)

    return passed, failed


if __name__ == '__main__':
    try:
        run_exectests(*sys.argv[1:])
    except BaseException:
        run_exectests(os.getcwd())
