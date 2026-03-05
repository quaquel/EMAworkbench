

# EMA Workbench Migration Guide

## Workbench 3.0
Version 3.0 is an extensive update of the workbench, drawing on developments in the python
ecosystem over the last decade. A large part of the API has remained the same, but there
are a number of backward incompatible changes. The minimum python version currently supported is 
Python 3.12. When upgrading, start with addressing the naming of parameters as outlined below. Next
ensure that basic experimentation (i.e., perform_experiments) works before moving on to updating
any optimization code. On the analysis side, the most important changes are in convergence
analysis (see the notebook example on this) and the removal of `threshold` from PRIM.

### naming of parameters
Perhaps the biggest change is that parameter names, outcomes names, constraint names, and model names now need to be valid 
python identifiers. That is, you cannot simply use a number or have spaces in the names of your parameters. For
those connecting to models in python, this change is not a big deal because their
parameter names were already valid python identifiers.

For Vensim users, the requirement that parameter names need to be valid python identifiers
might seem annoying, because the default behavior of the workbench is to map the parameter
name to a vensim variable. So, if you use underscores instead of spaces on the python side, you'll need underscores
inside vensim as well. To make life simpler for vensim users, therefore, by default, `VensimModel`
will process parameter names and replace underscores with spaces.

```python
# old
model = MyModel("flu")
model.uncertainties = [
    RealParameter("additional seasonal immune population fraction R1", 0, 0.5),
    RealParameter("additional seasonal immune population fraction R2", 0, 0.5,),
    RealParameter("fatality ratio region 1", 0.0001, 0.1,),
    RealParameter("fatality rate region 2", 0.0001, 0.1,)]

# new
model = MyModel()
model.uncertainties = [
    RealParameter("additional_seasonal_immune_population_fraction_R1", 0, 0.5),
    RealParameter("additional_seasonal_immune_population_fraction_R2", 0, 0.5,),
    RealParameter("fatality_ratio_region_1", 0.0001, 0.1,),
    RealParameter("fatality_rate_region_2", 0.0001, 0.1,)]
```

### overhaul of optimization
The support for many-objective optimization has been extensively updated, drawing on improvements
in `platypus-opt`. So, it is now possible to explicitly control the seed, run multiple seeds with
one command, and control the initial population with which the optimization starts (enabling restarts).
Moreover, the way in which optimization convergence is to be assessed has changed. The workbench
now will always store intermediate results in a tarball while `optimize` will return the final results
and any runtime convergence information coming from the algorithm. This means that `ArchiveLogger`
is no longer used and replaced with `filename`, `directory`, and `convergence_freq` keyword arguments. 

Other metrics (e.g., hypervolume) can no longer be calculated during the run but have to be done afterwards in a post processing step.
For this, the stored archives can be loaded using the new `load_archives` helper function. See the 
convergence analysis notebook example for full details on how to do convergence analysis. 

```python
# old
reference = Scenario("reference", b=0.4, q=2, mean=0.02, stdev=0.01)

convergence_metrics = [
    ArchiveLogger(
        "./data",
        [l.name for l in lake_model.levers],
        [o.name for o in lake_model.outcomes],
        base_filename="lake_model_dps_archive.tar.gz",
    ),
    EpsilonProgress(),
]

with MultiprocessingEvaluator(lake_model) as evaluator:
    results, convergence = evaluator.optimize(
        searchover="levers",
        nfe=100000,
        epsilons=[0.1] * len(lake_model.outcomes),
        reference=reference,
        convergence=convergence_metrics,
    )

# new
# note how we use Sample now instead of Scenario
reference = Sample("reference", b=0.4, q=2, mean=0.02, stdev=0.01)

random.seed(42)
seeds = [random.randint(0, 1000) for _ in range(5)] # we run the optimization for 5 different seeds

with MultiprocessingEvaluator(lake_model) as evaluator:
    # we run for 5 seeds, so optimize returns a list of
    # 5 tuples. Each tuple contains the results and any runtime info for one seed
    all_results = evaluator.optimize(
        searchover="levers",
        nfe=100000,
        convergence_freq=5000,
        epsilons=[0.1] * len(lake_model.outcomes),
        reference=reference,
        filename="lake_model_dps_archive.tar.gz",
        directory="./data/convergences",
        rng=seeds,
    )


```

### replacing `Policy` and `Scenario` classes with `Sample` class
In 2.x, the workbench made a distinction between a `Policy` and a `Scenario`. Under the hood, however, these objects were
identical. So, in 3.x, the distinction is dropped and, instead, it is called a Sample. We also introduced a new
`SampleCollection` class which is a collection of `Sample` instances. Likewise, `perform_experiments`, `optimize`, and 
`robust_optimize` have been updated to accept `Sample`, or an iterable returning `Sample` instances.

```python
# old
from ema_workbench import Scenario, Policy

scenario = Scenario("some scenario", a=1, b=2)
policy = Policy("some policy", x=1, y=2)

# new
from ema_workbench import Sample
scenario = Sample("some scenario", a=1, b=2)
policy = Sample("some policy", x=1, y=2)

```

### Fine-grained control over sampling
3.x offers more fine-grained control over sampling and samplers. First, there are new keyword arguments, 
`uncertainty_sampling_kwargs` and `lever_sampling_kwargs`, which are passed on to the various sampler classes. 
Second, there is more fine-grained control for combining scenarios and policies in `perform_experiments`, which 
now supports `full_factorial`, `cycle`, and `sample`. `full_factorial` is the default and runs all policies for all 
scenarios. `cycle` will repeat the shortest of the two deterministically until it matches the length of the longest. `sample` will 
upsample with replacement the shortest of the two to match the length of the longest. Last, the SALib samplers are 
now directly available from the `Samplers` enum.

### PRIM update 
The `threshold` argument has been removed from PRIM. This was causing confusion to users and was typically
not needed anyway. PRIM now always returns a box and any decision on whether the density is adequate is left
to the user. That is, a user should inspect the trade-off curve between coverage, density, and restricted dimensions and
assess which candidate box best balances these three objectives. 