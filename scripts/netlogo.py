import os
from ema_workbench import (Model, RealParameter, ScalarOutcome, ArrayOutcome,
                           SequentialEvaluator, MultiprocessingEvaluator)
from ema_workbench.connectors.netlogo import NetLogoModel

import jpype

if __name__ == "__main__":
    print(jpype.getDefaultJVMPath())

    # 1. Setting Up the NetLogo Interface:
    model = NetLogoModel('WolfSheepPredation',
                         wd="./netlogo-test",
                         netlogo_home="/opt/netlogo",
                         model_file="/opt/netlogo/app/models/Sample Models/Biology/Wolf Sheep Predation.nlogo",)
                         #jvm_path="/opt/netlogo/lib/runtime/lib/server/libjvm.so")

    # Model run setup
    model.run_length = 100
    model.replications = 5

    # 2. Define Model Uncertainties and Outcomes:

    # Specify the uncertainties
    model.uncertainties = [RealParameter('initial-number-sheep', 50, 100),
                           RealParameter('initial-number-wolves', 50, 100),
                           RealParameter('sheep-reproduce', 0.01, 0.1),
                           RealParameter('wolf-reproduce', 0.01, 0.1)]

    # Specify the outcomes
    model.outcomes = [ArrayOutcome('sheep'),
                      ArrayOutcome('wolves'),
                      ArrayOutcome('grass')]

    # 3. Run Experiments using the MultiprocessingEvaluator:

    with SequentialEvaluator(model) as evaluator:
        results = evaluator.perform_experiments(scenarios=25)

    experiments, outcomes = results

    # Do further analysis with the results as needed...
