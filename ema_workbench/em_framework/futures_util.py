import os
import shutil

from collections import defaultdict

from ..util import get_module_logger


_logger = get_module_logger(__name__)


def setup_working_directories(models, root_dir):
    """copies the working directory of each model to a process specific
    temporary directory and update the working directory of the model

    Parameters
    ----------
    models : list
    root_dir : str

    """

    # group models by working directory to avoid copying the same directory
    # multiple times
    wd_by_model = defaultdict(list)
    for model in models:
        try:
            wd = model.working_directory
        except AttributeError:
            pass
        else:
            wd_by_model[wd].append(model)

    # if the dict is not empty
    if wd_by_model:
        # make a directory with the process id as identifier
        tmpdir_name = f"tmp{os.getpid()}"
        tmpdir = os.path.join(root_dir, tmpdir_name)
        os.mkdir(tmpdir)

        _logger.debug(f"setting up working directory: {tmpdir}")

        for key, value in wd_by_model.items():
            # we need a sub directory in the process working directory
            # for each unique model working directory
            subdir = os.path.basename(os.path.normpath(key))
            new_wd = os.path.join(tmpdir, subdir)

            # the copy operation
            shutil.copytree(key, new_wd)

            for model in value:
                model.working_directory = new_wd
        return tmpdir
    else:
        return None
