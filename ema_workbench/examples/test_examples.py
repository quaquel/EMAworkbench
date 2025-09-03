import glob
import os
import sys
import traceback
from contextlib import contextmanager

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


def run_exectests(test_dir, log_path="exectests.log"):
    test_files = glob.glob(os.path.join(test_dir, "*.py"))
    test_files = sorted(ex for ex in test_files if ex[0] != "_")
    passed = []
    failed = []
    with open(log_path, "w") as f:
        with redirected_output(new_stdout=f, new_stderr=f):
            for fname in test_files:
                print(f">> Executing '{fname}'")
                try:
                    code = compile(set_backend + open(fname).read(), fname, "exec")
                    exec(code, {"__name__": "__main__"}, {})
                    passed.append(fname)
                except BaseException:
                    traceback.print_exc()
                    failed.append(fname)
                    pass

    print(f">> Passed {len(passed)}/{len(test_files)} tests: ")
    print("Passed: " + ", ".join(passed))
    print("Failed: " + ", ".join(failed))
    print(f"See {log_path} for details")

    return passed, failed


if __name__ == "__main__":
    try:
        run_exectests(*sys.argv[1:])
    except BaseException:
        run_exectests(os.getcwd())
