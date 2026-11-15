"""Created on 19 Nov 2018

@author: jhkwakkel
"""

import unittest

import matplotlib.pyplot as plt

from ema_workbench.analysis import regional_sa
from ema_workbench.util import EMAError
from test import utilities


class Test(unittest.TestCase):
    def test_plot_cdfs(self):
        x, outcomes = utilities.load_flu_data()
        y = outcomes["deceased_population_region_1"][:, -1] > 1000000

        regional_sa.plot_cdfs(x, y)
        regional_sa.plot_cdfs(x, y, ccdf=True)

        x = x.drop("scenario", axis=1)
        regional_sa.plot_cdfs(x, y, ccdf=True)

    def test_plot_cdfs_columns(self):
        """Regression test: the columns for which a cdf is plotted.

        Previously, a cdf was also plotted for the scenario column, and a
        DataFrame without a model or policy column raised a KeyError.
        """
        import numpy as np
        import pandas as pd

        def count_plots(fig):
            return sum(
                1
                for ax in fig.axes
                if ax.get_visible() and (ax.lines or ax.patches)
            )

        rng = np.random.default_rng(42)
        n = 200
        x = pd.DataFrame({"a": rng.random(n), "b": rng.random(n)})
        y = (x.a > 0.5).values

        # no scenario, policy, or model column
        self.assertEqual(count_plots(regional_sa.plot_cdfs(x, y)), 2)

        # as returned by perform_experiments, with a single policy and model
        experiments = x.assign(
            scenario=np.arange(n),
            policy=pd.Categorical(["p"] * n),
            model=pd.Categorical(["m"] * n),
        )
        self.assertEqual(count_plots(regional_sa.plot_cdfs(experiments, y)), 2)

        # with more than one policy, the policy is included
        experiments["policy"] = pd.Categorical(rng.choice(["p", "q"], n))
        self.assertEqual(count_plots(regional_sa.plot_cdfs(experiments, y)), 3)

        # like in the other scenario discovery techniques, columns with a
        # single value are dropped
        experiments["constant"] = 1.0
        self.assertEqual(count_plots(regional_sa.plot_cdfs(experiments, y)), 3)

        with self.assertRaises(EMAError):
            regional_sa.plot_cdfs(experiments[["scenario", "model", "constant"]], y)
        plt.close("all")

    def test_plot__individual_cdf(self):
        x, outcomes = utilities.load_flu_data()
        y = outcomes["deceased_population_region_1"][:, -1] > 1000000

        fig, ax = plt.subplots()
        unc = "fatality_ratio_region_1"

        regional_sa.plot_individual_cdf(
            ax,
            unc,
            x[unc],
            y,
            discrete=False,
            legend=True,
            xticklabels_on=True,
            yticklabels_on=True,
        )

        fig, ax = plt.subplots()
        unc = "model"

        regional_sa.plot_individual_cdf(
            ax,
            unc,
            x[unc],
            y,
            discrete=True,
            legend=True,
            xticklabels_on=True,
            yticklabels_on=True,
        )


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
