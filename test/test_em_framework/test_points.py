

import unittest

from ema_workbench.em_framework.util import NamedObject
from ema_workbench.em_framework import points

class TestCases(unittest.TestCase):

    def test_experiment_gemerator(self):
        scenarios = [NamedObject("scen_1"), NamedObject("scen_2")]
        model_structures = [NamedObject("model")]
        policies = [NamedObject("1"), NamedObject("2"), NamedObject("3")]

        experiments = points.experiment_generator(scenarios,
                                                  model_structures,
                                                  policies,
                                                  combine='factorial')
        experiments = [e for e in experiments]
        self.assertEqual(len(experiments), 6, ("wrong number of experiments "
                                               "for factorial"))

        experiments = points.experiment_generator(scenarios,
                                                  model_structures,
                                                  policies,
                                                  combine='sample')
        experiments = [e for e in experiments]
        self.assertEqual(len(experiments), 3, ("wrong number of experiments "
                                               "for zipover"))

        with self.assertRaises(ValueError):
            experiments = points.experiment_generator(scenarios,
                                                    model_structures,
                                        policies, combine='adf')
            _ = [e for e in experiments]


    # def test_experiment_generator(self):
    #     sampler = LHSSampler()
    #
    #     shared_abc_1 = parameters.RealParameter("shared ab 1", 0, 1)
    #     shared_abc_2 = parameters.RealParameter("shared ab 2", 0, 1)
    #     unique_a = parameters.RealParameter("unique a ", 0, 1)
    #     unique_b = parameters.RealParameter("unique b ", 0, 1)
    #     uncertainties = [shared_abc_1, shared_abc_2, unique_a, unique_b]
    #     designs = sampler.generate_designs(uncertainties, 10)
    #     designs.kind = Scenario
    #
    #     # everything shared
    #     model_a = Model("A", mock.Mock())
    #     model_b = Model("B", mock.Mock())
    #
    #     model_a.uncertainties = [shared_abc_1, shared_abc_2, unique_a]
    #     model_b.uncertainties = [shared_abc_1, shared_abc_2, unique_b]
    #     model_structures = [model_a, model_b]
    #
    #     policies = [Policy('policy 1'),
    #                 Policy('policy 2'),
    #                 Policy('policy 3')]
    #
    #     gen = parameters.experiment_generator(designs, model_structures,
    #                                           policies)
    #
    #     experiments = []
    #     for entry in gen:
    #         experiments.append(entry)
    #     self.assertEqual(len(experiments), 2 * 3 * 10)


if __name__ == "__main__":
    unittest.main()
