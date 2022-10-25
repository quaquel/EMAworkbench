# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- Release notes generated using configuration in .github/release.yml at master -->

## 2.2.0
### Highlights
With the 2.2 release, the EMAworkbench can now connect to [Vadere](https://www.vadere.org/) models on pedestrian dynamics. When inspecting a Prim Box peeling trajectory, multiple points on the peeling trajectory can be inspected simultaneously by inputting a list of integers into [`PrimBox.inspect()`](https://emaworkbench.readthedocs.io/en/latest/ema_documentation/analysis/prim.html#ema_workbench.analysis.prim.PrimBox.inspect).

When running experiments with multiprocessing using the [`MultiprocessingEvaluator`](https://emaworkbench.readthedocs.io/en/latest/ema_documentation/em_framework/evaluators.html#ema_workbench.em_framework.evaluators.MultiprocessingEvaluator), the number of processes can now be controlled using a negative integer as input for `n_processes` (for example, `-2` on a 12-thread CPU results in 10 threads used). Also, it will now default to max. 61 processes on windows machines due to limitations inherent in Windows in dealing with higher processor counts.  Code quality, CI, and error reporting also have been improved. And finally, generating these release notes is now automated.

### What's Changed
#### 🎉 New features added
* Vadere model connector by @floristevito in https://github.com/quaquel/EMAworkbench/pull/145

#### 🛠 Enhancements made
* Improve code quality with static analysis by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/119
* prim.py: Make `PrimBox.peeling_trajectory["id"]` int instead of float by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/121
* analysis: Allow optional annotation of plot_tradeoff graphs by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/123
* evaluators.py: Allow MultiprocessingEvaluator to initialize with cpu_count minus N processes by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/140
* `PrimBox.inspect()` now can also take a list of integers (aside from a single int) to inspect multiple points at once by @quaquel in https://github.com/quaquel/EMAworkbench/commit/6d83a6c33442ad4dce0974a384b03a225aaf830d (see also issue https://github.com/quaquel/EMAworkbench/issues/124)

#### 🐛 Bugs fixed
* fixed typo in lake_model.py by @JeffreyDillonLyons in https://github.com/quaquel/EMAworkbench/pull/136

#### 📜 Documentation improvements
* Docs: Installation.rst: Add how to install master or custom branch by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/122
* Docs: Replace all http links with secure https URLs by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/134
* Maintain release notes at CHANGELOG.md and include them in Readthedocs by @quaquel in https://github.com/quaquel/EMAworkbench/commit/ebdbc9f5c77693fc75911ead472b420065dfe2aa
* Fix badge links in readme by @quaquel in https://github.com/quaquel/EMAworkbench/commit/28569bdcb149c070c329589969179be354b879ec

#### 🔧 Maintenance
* feature_scoring: fix Regressor criterion depreciation by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/125
* feature_scoring.py: Change `max_features` in get_rf_feature_scores to `"sqrt"` by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/129
* CI: Use Pytest instead of Nose, update default build to Python 3.10 by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/131
* Release CI: Only upload packages if on main repo by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/132
* CI: Split off flake8 linting in a separate job by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/133
* CI: Add weekly scheduled jobs and manual trigger by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/137
* setup.py: Add `project_urls` for documentation and issue tracker links by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/142
* set scikit-learn requirement >= 1.0.0 by @rhysits in https://github.com/quaquel/EMAworkbench/pull/144
* Create release.yml file for automatic release notes generation by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/152
* instantiating an Evaluator without one or more AbstractModel instances now raises a type error by @quaquel in https://github.com/quaquel/EMAworkbench/commit/a83533aa8166ca2414137cdfc3125a53ee3697ec
* removes depreciated DataFrame.append by replacing it with DataFrame.concat (see the conversation on issue https://github.com/quaquel/EMAworkbench/issues/126):
  * from feature scoring by @quaquel in https://github.com/quaquel/EMAworkbench/commit/8b8bfe41733e49b75c01e34b75563e0a6d5b4024
  *  from logistic_regression.py by @quaquel in https://github.com/quaquel/EMAworkbench/commit/255e3d6d9639dfe6fd4e797e1c63d59ba0522c2d
* removes NumPy datatypes deprecated in 1.20 by @quaquel in https://github.com/quaquel/EMAworkbench/commit/e8fbf6fc64f14b7c7220fa4d3fc976c42d3757eb
* replace deprecated scipy.stats.kde with scipy.stats by @quaquel in https://github.com/quaquel/EMAworkbench/commit/b5a9ca967740e74d503281018e88d6b28e74a27d




### New Contributors
* @JeffreyDillonLyons made their first contribution in https://github.com/quaquel/EMAworkbench/pull/136
* @rhysits made their first contribution in https://github.com/quaquel/EMAworkbench/pull/144
* @floristevito made their first contribution in https://github.com/quaquel/EMAworkbench/pull/145

**Full Changelog**: https://github.com/quaquel/EMAworkbench/compare/2.1.2...2.2.0
