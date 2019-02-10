
import os
import sys
import traceback
import math
from itertools import zip_longest

from .evaluators import BaseEvaluator
from ..util import ema_logging
from .parameters import experiment_generator, Case
from ..util.ema_exceptions import EMAError, CaseError

from ema_workbench.util import get_module_logger
_logger = get_module_logger(__name__)

from dask.distributed import Client, as_completed, get_worker

def store_model_on_worker(name, model):
	worker = get_worker()
	if not hasattr(worker, '_ema_models'):
		worker._ema_models = {}
	worker._ema_models[name] = model


def run_experiment_on_worker(experiment):
	'''Run a single experiment on a dask worker.

	This code makes sure that model is initialized correctly.

	Parameters
	----------
	experiment : Case

	Returns
	-------
	experiment_id: int
	result : dict

	Raises
	------
	EMAError
		if the model instance raises an EMA error, these are reraised.
	Exception
		Catch all for all other exceptions being raised by the model.
		These are reraised.

	'''
	worker = get_worker()

	model_name = experiment.model_name
	model = worker._ema_models[model_name]
	policy = experiment.policy.copy()

	scenario = experiment.scenario
	try:
		model.run_model(scenario, policy)
	except CaseError as e:
		ema_logging.warning(str(e))
	except Exception as e:
		ema_logging.exception(str(e))
		try:
			model.cleanup()
		except Exception:
			raise e

		exception = traceback.print_exc()
		if exception:
			sys.stderr.write(exception)
			sys.stderr.write("\n")

		errortype = type(e).__name__
		raise EMAError(("exception in run_model"
						"\nCaused by: {}: {}".format(errortype, str(e))))

	outcomes = model.outcomes_output
	model.reset_model()

	return experiment.experiment_id, outcomes.copy()

def run_experiments_on_worker(experiments):
	"""
	Run multiple experiments in a batch on one worker.

	Sending a batch of experiments cuts down on the number of communications
	required between scheduler and worker processes.

	Parameters
	----------
	experiments : Iterable of Case

	Returns
	-------
	tuple
		The results from `run_experiment_on_worker`
	"""
	return tuple(run_experiment_on_worker(experiment) for experiment in experiments if experiment is not None)


def grouper(iterable, n, fillvalue=None):
	"Collect data into fixed-length chunks or blocks"
	# grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
	args = [iter(iterable)] * n
	return zip_longest(*args, fillvalue=fillvalue)



class DistributedEvaluator(BaseEvaluator):
	"""Evaluator using dask.distributed

    Parameters
    ----------
    msis : collection of models
    client : distributed.Client (optional)
             A client can be provided. If one is not provided, a default Client
             will be created.
    batch_size : int (optional)
                 The number of experiment to batch together when pushing tasks to distributed workers.
                 If not given, the first call to evaluate_experiments will make a reasonable guess that
                 will allocate batches so that there are about 10 tasks per worker.  This may or may not
                 be efficient.
	max_n_workers : int (default 32)
	                The maximum number of workers that will be created for a default Client.  If the number
	                of cores available is smaller than this number, fewer workers will be spawned.

	"""

	_default_client = None

	def __init__(self, msis, *, client=None, batch_size=None, max_n_workers=32):
		super().__init__(msis, )

		# Initialize a default dask.distributed client if one is not given
		if client is None:
			if type(self)._default_client is None:
				import multiprocessing
				n_workers = min(multiprocessing.cpu_count(), max_n_workers)
				type(self)._default_client = Client(
					n_workers=n_workers,
					threads_per_worker=1,
				)
			client = type(self)._default_client

		self.client = client
		self.batch_size = batch_size

	def initialize(self):
		self.broadcast_models_to_workers()

	def finalize(self):
		pass

	def broadcast_models_to_workers(self):
		for msi in self._msis:
			self.client.run(store_model_on_worker, msi.name, msi)

	def evaluate_experiments(self, scenarios, policies, callback, zip_over=None):
		_logger.debug("evaluating experiments asynchronously")

		ex_gen = experiment_generator(scenarios, self._msis, policies, zip_over)

		cwd = os.getcwd()

		log_message = ('storing scenario %s for policy %s on model %s')

		experiments = {
			experiment.experiment_id: experiment
			for experiment in ex_gen
		}

		if self.batch_size is None:
			# make a guess at a good batch size if one was not given
			n_workers = len(self.client.scheduler_info()['workers'])
			n_experiments = len(experiments)
			self.batch_size = math.ceil(n_experiments / n_workers / 10 )

		# Experiments are sent to workers in batches, as the task-scheduler overhead is high for quick-running models.
		batches = grouper(experiments.values(), self.batch_size)
		outcomes = self.client.map(run_experiments_on_worker, batches)

		_logger.debug("receiving experiments asynchronously")

		for future, result_batch in as_completed(outcomes, with_results=True):
			for (experiment_id, outcome) in result_batch:
				experiment = experiments[experiment_id]
				_logger.debug(
					log_message,
					experiment.scenario.name,
					experiment.policy.name,
					experiment.model_name,
				)
				callback(experiment, outcome)

		os.chdir(cwd)

		_logger.debug("completed evaluate_experiments")
