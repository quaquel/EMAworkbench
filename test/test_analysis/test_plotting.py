"""
Created on 22 jul. 2012

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
"""
import unittest

import matplotlib.pyplot as plt
import numpy as np

from ema_workbench.analysis.b_and_w_plotting import set_fig_to_bw
from ema_workbench.analysis.plotting import *
from ema_workbench.analysis.plotting_util import (make_continuous_grouping_specifiers,
                                                  filter_scalar_outcomes, group_results,
                                                  Density, PlotType)
from test import utilities
from ema_workbench.util.ema_exceptions import EMAError
# from ema_workbench.em_framework.outcomes import ScalarOutcome, ArrayOutcome


# don't run these tests using nosetest
# __test__ = False

class TestPlotting(unittest.TestCase):

    def test_make_continuous_grouping_specifiers(self):
        n_groups = 10
        array = np.linspace(0, 100, 50)
        categories = make_continuous_grouping_specifiers(array,
                                                         nr_of_groups=n_groups)

        limits = np.linspace(0, 100, n_groups+1)

        for i, entry in enumerate(categories):
            a = tuple(limits[i:i+2])
            self.assertEqual(entry, a)

    def test_filter_scalar_outcomes(self):
        outcomes = {}
        for entry in ['a', 'b', 'c']:
            outcomes[entry] = np.random.rand(10, 100)
        for entry in ['d','e','f']:
            outcomes[entry] = np.random.rand(10)
        outcomes = filter_scalar_outcomes(outcomes)

        for entry in ['a', 'b', 'c']:
            self.assertIn(entry, outcomes.keys())

    def test_group_results(self):
        results = utilities.load_eng_trans_data()
        experiments, outcomes = results

        # test indices
        groups = {'set1':np.arange(0,11),
                  'set2':np.arange(11,25),
                  'set3':np.arange(25,experiments.shape[0])}
        groups = group_results(experiments, outcomes,
                               group_by='index',
                               grouping_specifiers=groups.values(),
                               grouping_labels= groups.keys())
        total_data = 0
        for value in groups.values():
            total_data += value[0].shape[0]
        print(experiments.shape[0], total_data)

        # test continuous parameter type
        array = experiments['average planning and construction period T1']
        grouping_specifiers = make_continuous_grouping_specifiers(array, nr_of_groups=5)
        groups = group_results(experiments, outcomes,
                               group_by='average planning and construction period T1',
                               grouping_specifiers=grouping_specifiers,
                               grouping_labels = [str(entry) for entry in grouping_specifiers])
        total_data = 0
        for value in groups.values():
            total_data += value[0].shape[0]
        print(experiments.shape[0], total_data)

        # test integer type
        array = experiments['seed PR T1']
        grouping_specifiers = make_continuous_grouping_specifiers(array, nr_of_groups=10)
        groups = group_results(experiments, outcomes,
                               group_by='seed PR T1',
                               grouping_specifiers=grouping_specifiers,
                               grouping_labels = [str(entry) for entry in grouping_specifiers])
        total_data = 0
        for value in groups.values():
            total_data += value[0].shape[0]
        print(experiments.shape[0], total_data)

        # test categorical type
        grouping_specifiers = set(experiments["policy"])
        groups = group_results(experiments, outcomes,
                           group_by='policy',
                           grouping_specifiers=grouping_specifiers,
                           grouping_labels = [str(entry) for entry in grouping_specifiers])
        total_data = 0
        for value in groups.values():
            total_data += value[0].shape[0]
        print(experiments.shape[0], total_data)

    def test_lines(self):
        experiments, outcomes = utilities.load_eng_trans_data()

        lines(experiments, outcomes,
              outcomes_to_show="total fraction new technologies",
              experiments_to_show=np.arange(0, 600, 20),
              group_by='policy',
              grouping_specifiers='basic policy'
              )
        lines(experiments, outcomes,
              experiments_to_show=np.arange(0, 600, 2),
              group_by='policy',
              density=Density.HIST
              )
        lines(experiments, outcomes,
              experiments_to_show=np.arange(0, 600, 2),
              group_by='policy',
              density=Density.KDE
              )
        lines(experiments, outcomes,
              experiments_to_show=np.arange(0, 600, 2),
              group_by='policy',
              density=Density.BOXPLOT
              )
        lines(experiments, outcomes,
              experiments_to_show=np.arange(0, 600, 2),
              group_by='policy',
              density=Density.VIOLIN
              )
        lines(experiments, outcomes,
              group_by='index',
              grouping_specifiers = {"blaat": np.arange(1, 100, 2)},
              density=Density.KDE,
              )

        lines(experiments, outcomes,
              experiments_to_show=np.arange(0, 600, 30),
              group_by='policy',
              density=Density.KDE,
              show_envelope=True,
              grouping_specifiers=['no policy', 'adaptive policy']
              )
        lines(experiments, outcomes,
              experiments_to_show=np.arange(0, 600, 30),
              group_by='policy',
              density=Density.HIST,
              show_envelope=True,
              grouping_specifiers=['no policy', 'adaptive policy']
              )
        lines(experiments, outcomes,
              experiments_to_show=np.arange(0, 600, 30),
              group_by='policy',
              density=Density.BOXPLOT,
              show_envelope=True,
              grouping_specifiers=['no policy', 'adaptive policy']
              )
        lines(experiments, outcomes,
              experiments_to_show=np.arange(0, 600, 30),
              group_by='policy',
              density=Density.VIOLIN,
              show_envelope=True,
              grouping_specifiers=['no policy', 'adaptive policy']
              )

        plt.draw()
        plt.close('all')

        lines(experiments, outcomes,
              experiments_to_show=np.arange(0, 600, 30),
              group_by='policy',
              density=Density.KDE,
              show_envelope=True,
              grouping_specifiers=['no policy', 'adaptive policy'],
              log=True
              )
        lines(experiments, outcomes,
              experiments_to_show=np.arange(0, 600, 30),
              group_by='policy',
              density=Density.HIST,
              show_envelope=True,
              grouping_specifiers=['no policy', 'adaptive policy'],
              log=True
              )
        lines(experiments, outcomes,
              experiments_to_show=np.arange(0, 600, 30),
              group_by='policy',
              density=Density.BOXPLOT,
              show_envelope=True,
              grouping_specifiers=['no policy', 'adaptive policy'],
              log=True
              )
        lines(experiments, outcomes,
              experiments_to_show=np.arange(0, 600, 30),
              group_by='policy',
              density=Density.VIOLIN,
              show_envelope=True,
              grouping_specifiers=['no policy', 'adaptive policy'],
              log=True
              )

        plt.draw()
        plt.close('all')

        set_fig_to_bw(lines(experiments, outcomes,
                 experiments_to_show=np.arange(0, 600, 20),
                 group_by='policy',
                 density=Density.KDE
                 )[0])

        new_outcomes = {}
        for key, value in outcomes.items():
            new_outcomes[key] = value[0:20, :]
        experiments = experiments[0:20]

        #no grouping, with density
        set_fig_to_bw(lines(experiments, new_outcomes, density=Density.KDE)[0])
        set_fig_to_bw(lines(experiments, new_outcomes, density=Density.HIST)[0])
        set_fig_to_bw(lines(experiments, new_outcomes, density=Density.BOXPLOT)[0])
        set_fig_to_bw(lines(experiments, new_outcomes, density=Density.VIOLIN)[0])

        # grouping and density
        set_fig_to_bw(lines(experiments, new_outcomes,
              group_by='policy',
              density=Density.KDE)[0])

        # grouping, density as histograms
        # grouping and density
        set_fig_to_bw(lines(experiments, new_outcomes,
              group_by='policy',
              density=Density.HIST,
              legend=False)[0])

        plt.draw()
        plt.close('all')

    def test_envelopes(self):
        # TODO:: should iterate over the density enum to automatically
        # test all defined densities

        experiments, outcomes = utilities.load_eng_trans_data()

        #testing titles
        envelopes(experiments, outcomes,
                  density=None,
                  titles=None)
        envelopes(experiments, outcomes,
              density=None,
              titles={})
        envelopes(experiments, outcomes,
              density=None,
              titles={'total fraction new technologies': 'a'})

        plt.draw()
        plt.close('all')

        #testing ylabels
        envelopes(experiments, outcomes,
                  density=None,
                  ylabels=None)
        envelopes(experiments, outcomes,
              density=None,
              ylabels={})
        envelopes(experiments, outcomes,
              density=None,
              ylabels={'total fraction new technologies': 'a'})

        plt.draw()
        plt.close('all')


        #no grouping no density
        envelopes(experiments, outcomes,
                  titles=None)
        set_fig_to_bw(envelopes(experiments, outcomes, density=None)[0])

        plt.draw()
        plt.close('all')

        #no grouping, with density
        envelopes(experiments, outcomes, density=Density.KDE)
        envelopes(experiments, outcomes, density=Density.HIST)
        envelopes(experiments, outcomes, density=Density.BOXPLOT)
        envelopes(experiments, outcomes, density=Density.VIOLIN)
        envelopes(experiments, outcomes, density=Density.BOXENPLOT)

        with self.assertRaises(EMAError):
            envelopes(experiments, outcomes, density='undefined')

        set_fig_to_bw(envelopes(experiments, outcomes, density=Density.VIOLIN)[0])

        plt.draw()
        plt.close('all')

        # grouping and density kde
        envelopes(experiments, outcomes,
                  group_by='policy',
                  density=Density.VIOLIN)
        envelopes(experiments, outcomes,
                  group_by='policy',
                  density=Density.BOXPLOT)
        envelopes(experiments, outcomes,
                  group_by='policy',
                  density=Density.KDE,
                  grouping_specifiers=['no policy', 'adaptive policy'])
        envelopes(experiments, outcomes,
                  group_by='policy',
                  density=Density.BOXPLOT,
                  grouping_specifiers=['no policy', 'adaptive policy'])
        envelopes(experiments, outcomes,
                  group_by='policy',
                  density=Density.KDE)
        envelopes(experiments, outcomes,
                  group_by='policy',
                  density=Density.BOXENPLOT)

        plt.draw()
        plt.close('all')

        envelopes(experiments, outcomes,
                  group_by='policy',
                  density=Density.VIOLIN)
        envelopes(experiments, outcomes,
                  group_by='policy',
                  density=Density.BOXPLOT)
        envelopes(experiments, outcomes,
                  group_by='policy',
                  density=Density.KDE)
        envelopes(experiments, outcomes,
                  group_by='policy',
                  density=Density.HIST)

        plt.draw()
        plt.close('all')

        envelopes(experiments, outcomes,
                  group_by='policy',
                  density=Density.VIOLIN,
                  log=True)
        envelopes(experiments, outcomes,
                  group_by='policy',
                  density=Density.BOXPLOT,
                  log=True)
        envelopes(experiments, outcomes,
                  group_by='policy',
                  density=Density.KDE,
                  log=True)
        envelopes(experiments, outcomes,
                  group_by='policy',
                  density=Density.HIST,
                  log=True)
        envelopes(experiments, outcomes,
                  group_by='policy',
                  density=Density.BOXENPLOT,
                  log=True)

        plt.draw()
        plt.close('all')

        # grouping and density hist
        envelopes(experiments, outcomes,
                  group_by='policy',
                  density=Density.HIST)
        envelopes(experiments, outcomes,
                  group_by='policy',
                  density=Density.HIST)

        set_fig_to_bw(envelopes(experiments, outcomes,
                  group_by='policy',
                  density=Density.KDE)[0])

        # grouping and density
        envelopes(experiments, outcomes,
                  group_by='policy',
                  density=Density.KDE,
                  fill=True)
        set_fig_to_bw(envelopes(experiments, outcomes,
                  group_by='policy',
                  density=Density.KDE,
                  fill=True)[0])

        plt.draw()
        plt.close('all')

    def test_kde_over_time(self):
        experiments, outcomes = utilities.load_eng_trans_data()

        kde_over_time(experiments, outcomes, log=False)
        kde_over_time(experiments, outcomes, log=True)
        kde_over_time(experiments, outcomes, group_by='policy',
                      grouping_specifiers=['no policy', 'adaptive policy'])
        plt.draw()
        plt.close('all')


    def test_multiple_densities(self):
        experiments, outcomes = utilities.load_eng_trans_data()
        ooi = 'total fraction new technologies'

        multiple_densities(experiments, outcomes,
                      group_by="policy",
                      points_in_time = [2010])
        multiple_densities(experiments, outcomes,
                      outcomes_to_show=ooi,
                      group_by="policy",
                      points_in_time = [2010])
        multiple_densities(experiments, outcomes,
                      outcomes_to_show=ooi,
                      group_by="policy",
                      points_in_time = [2010, 2100])
        multiple_densities(experiments, outcomes,
                      outcomes_to_show=ooi,
                      group_by="policy",
                      points_in_time = [2010, 2050, 2100])
        multiple_densities(experiments, outcomes,
                      outcomes_to_show=ooi,
                      group_by="policy",
                      points_in_time = [2010, 2020, 2050, 2080])
        multiple_densities(experiments, outcomes,
                      outcomes_to_show=ooi,
                      group_by="policy",
                      points_in_time = [2010, 2020, 2040, 2060, 2100])

        multiple_densities(experiments, outcomes,
                      outcomes_to_show=ooi,
                      group_by="policy",
                      points_in_time = [2010, 2020, 2040, 2060, 2080, 2100],
                      plot_type=PlotType.ENV_LIN,
                      density=Density.KDE,
                      experiments_to_show=[1, 2, 10])
        multiple_densities(experiments, outcomes,
                      outcomes_to_show=ooi,
                      group_by="policy",
                      points_in_time = [2010, 2020, 2040, 2060, 2080, 2100],
                      plot_type=PlotType.ENV_LIN,
                      density=Density.HIST,
                      experiments_to_show=[1, 2, 10])
        multiple_densities(experiments, outcomes,
                      outcomes_to_show=ooi,
                      group_by="policy",
                      points_in_time = [2010, 2020, 2040, 2060, 2080, 2100],
                      plot_type=PlotType.ENV_LIN,
                      density=Density.BOXPLOT,
                      experiments_to_show=[1, 2, 10])
        multiple_densities(experiments, outcomes,
                      outcomes_to_show=ooi,
                      group_by="policy",
                      points_in_time = [2010, 2020, 2040, 2060, 2080, 2100],
                      plot_type=PlotType.ENV_LIN,
                      density=Density.VIOLIN,
                      experiments_to_show=[1, 2, 10])

        plt.draw()
        plt.close('all')

        multiple_densities(experiments, outcomes,
                      outcomes_to_show=ooi,
                      group_by="policy",
                      points_in_time=[2010, 2020, 2040, 2060, 2080, 2100],
                      plot_type=PlotType.ENV_LIN,
                      density=Density.KDE,
                      experiments_to_show=[1, 2, 10],
                      log=True)
        multiple_densities(experiments, outcomes,
                      outcomes_to_show=ooi,
                      group_by="policy",
                      points_in_time=[2010, 2020, 2040, 2060, 2080, 2100],
                      plot_type=PlotType.ENV_LIN,
                      density=Density.HIST,
                      experiments_to_show=[1,2,10],
                      log=True)
        multiple_densities(experiments, outcomes,
                      outcomes_to_show=ooi,
                      group_by="policy",
                      points_in_time=[2010, 2020, 2040, 2060, 2080, 2100],
                      plot_type=PlotType.ENV_LIN,
                      density=Density.BOXPLOT,
                      experiments_to_show=[1, 2, 10],
                      log=True)
        multiple_densities(experiments, outcomes,
                      outcomes_to_show=ooi,
                      group_by="policy",
                      points_in_time=[2010, 2020, 2040, 2060, 2080, 2100],
                      plot_type=PlotType.ENV_LIN,
                      density=Density.VIOLIN,
                      experiments_to_show=[1, 2, 10],
                      log=True)

        plt.draw()
        plt.close('all')


if __name__ == '__main__':
    suite = unittest.TestSuite()
    # suite.addTest(TestPlotting("test_make_continuous_grouping_specifiers"))
    # suite.addTest(TestPlotting("test_filter_scalar_outcomes"))
    # suite.addTest(TestPlotting("test_group_results"))
    suite.addTest(TestPlotting("test_lines"))
    # suite.addTest(TestPlotting("test_envelopes"))
    # suite.addTest(TestPlotting("test_kde_over_time"))
    # suite.addTest(TestPlotting("test_multiple_densities"))
    runner = unittest.TextTestRunner()
    runner.run(suite)

    # unittest.main()