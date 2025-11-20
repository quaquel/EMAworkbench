

# EMA Workbench Migration Guide

## Workbench 3.0

Version 3.0 is an extensive update of the workbench, drawing in developments in the python
ecosystem over the last decade. A large part of the API has remained the same, but there
are a number of backward incompatible changes.

### naming of parameters
Perhaps the biggest change is that parameter names now need to be valid python identifiers.
That is, you cannot simply use a number or have spaces in the names of your parameters. For
those connecting to models in python, this change is not a big deal because their
parameter names were already valid python identifiers.

For Vensim users, the requirement that parameter names need to be valid python identifiers
might seem annoying, because the default behavior of the workbench is to make  the parameter
name to a vensim variable. So, you have underscored on the python side, you'll need underscores
in vensim as well. To make life simpler for vensim users, therefore, by default, `VensimModel`
will process parameter names and replace underscores with spaces.

```python
# old
model = VensimModel(
    "flu", wd=r"./models/flu", model_file=r"FLUvensimV1basecase.vpm"
)
model.uncertainties = [
    RealParameter("additional seasonal immune population fraction R1", 0, 0.5),
    RealParameter("additional seasonal immune population fraction R2", 0, 0.5,),
    RealParameter("fatality ratio region 1", 0.0001, 0.1,),
    RealParameter("fatality rate region 2", 0.0001, 0.1,)]

# new
model = VensimModel(
    "flu", wd=r"./models/flu", model_file=r"FLUvensimV1basecase.vpm"
)
model.uncertainties = [
    RealParameter("additional_seasonal_immune_population_fraction_R1", 0, 0.5),
    RealParameter("additional_seasonal_immune_population_fraction_R2", 0, 0.5,),
    RealParameter("fatality_ratio_region_1", 0.0001, 0.1,),
    RealParameter("fatality_rate_region_2", 0.0001, 0.1,)]
```

### overhaul of optimization
The support for many-objective optimization has been extensively updated, drawing on improvements
in `platypus-opt`. So is it now possible to explicitly control the seed, run multiple seeds with
one command, and control the initial population with which the optimization starts (enabling restarts).
Moreover, the way in which optimization convergence is to be assessed has changed. The workbench
now will allways store intermediate results in a tarball while `optimize` will return the final results
and any runtime convergence information coming from the algorithm. Other metrics (e.g., hypervolume) can
no longer be calculated during the run but have to be done afterwards in a post processing step.

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
reference = Sample("reference", b=0.4, q=2, mean=0.02, stdev=0.01)

random.seed(42)
seeds = [random.randint(0, 1000) for _ in range(5)] # we run the optimization for 5 different seeds

with MultiprocessingEvaluator(lake_model) as evaluator:
    results, runtime_info = evaluator.optimize(
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



### Finegrained control over sampling
