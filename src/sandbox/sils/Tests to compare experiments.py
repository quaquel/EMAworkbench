def test_experiment_from_cases_vs_stored_on_non_game():
#    CONSTRUCTING THE ENSEMBLE AND SAVING THE RESULTS
    EMAlogging.log_to_stderr(EMAlogging.DEBUG)
    results = load_results(r'base.cPickle')

    runs = [526,781,911,988,10,780,740,943,573,991]
    runs = sorted(runs)
    VOI = 'relative market price'
    beh_no = 1
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    model_location = r'.\Models\starting\Metals EMA.vpm'
    vensim.load_model(model_location)
    
    for beh_no in range(1): 
        results_of_interest = experiment_settings(results,runs,VOI)
        cases_of_interest = experiments_to_cases(results_of_interest[0])
       
        behaviour_int = results_of_interest[1][VOI]
        
        case = cases_of_interest[beh_no]
        behavior = behaviour_int[beh_no]

        case = copy.deepcopy(case)
        set_lookups(case)
        
        for key,value in case.items():
            vensim.set_value(key,value)
        vensim.run_simulation('Test.vdf')

        interval_series = vensim.get_data(r'C:\workspace\EMA-workbench\src\sandbox\sils\Models\starting\Test.vdf',VOI)
        ax1.plot(interval_series, label='new', c='b')
        ax1.plot(behavior, label='original', c='r')
        ax1.legend(loc='best')
    plt.show()

def test_experiment_from_cases_vs_stored_1():
#    CONSTRUCTING THE ENSEMBLE AND SAVING THE RESULTS
    EMAlogging.log_to_stderr(EMAlogging.DEBUG)
    results = load_results(r'base.cPickle')

    runs = [526,781,911,988,10,780,740,943,573,991]
    runs = sorted(runs)
    VOI = 'relative market price'
    beh_no = 0
    
    results_of_interest = experiment_settings(results,runs,VOI)
    cases_of_interest = experiments_to_cases(results_of_interest[0])
   
    behaviour_int = results_of_interest[1][VOI]
    
    case = cases_of_interest[beh_no]
    behavior = behaviour_int[beh_no]
    
    model_location = r'C:\workspace\EMA-workbench\src\sandbox\sils\Models\Consecutive\Metals EMA with switches for consecutive.vpm'
    vensim.load_model(model_location)

    case = copy.deepcopy(case)
    set_lookups(case)
    
    for key,value in case.items():
        vensim.set_value(key,value)
    
    # Initiate the model to be run in game mode.
    venDLL.command('SIMULATE>RUNNAME|Test|O')
    venDLL.command('GAME>GAMEINTERVAL|'+str(vensim.get_val('FINAL TIME')))
    venDLL.command("MENU>GAME")
        
    venDLL.command('GAME>GAMEON')
    venDLL.command('GAME>ENDGAME')
    
    print vensim.get_val('TIME')
    
    interval_series = vensim.get_data(r'C:\workspace\EMA-workbench\src\sandbox\sils\Models\Consecutive\Test.vdf',VOI)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    ax1.plot(interval_series, label='new')
    ax1.legend()
    ax2.plot(behavior, label='original')
    ax2.legend()
    plt.show()

def test_experiment_from_cases_vs_stored():
#    CONSTRUCTING THE ENSEMBLE AND SAVING THE RESULTS
    EMAlogging.log_to_stderr(EMAlogging.DEBUG)
    results = load_results(r'base.cPickle')

    runs = [526,781,911,988,10,780,740,943,573,991]
    runs = sorted(runs)
    VOI = 'relative market price'
    beh_no = 1
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    model_location = r'.\Models\Base\Metals EMA.vpm'
    vensim.load_model(model_location)


    
    for beh_no in range(1): 
        results_of_interest = experiment_settings(results,runs,VOI)
        cases_of_interest = experiments_to_cases(results_of_interest[0])
       
        behaviour_int = results_of_interest[1][VOI]
        
        case = cases_of_interest[beh_no]
        behavior = behaviour_int[beh_no]

        case = copy.deepcopy(case)
        set_lookups(case)
        for key,value in case.items():
            vensim.set_value(key,value)
        
        # Initiate the model to be run in game mode.
        venDLL.command('SIMULATE>RUNNAME|Test|O')
        venDLL.command('GAME>GAMEINTERVAL|'+str(vensim.get_val('FINAL TIME')))
        venDLL.command("MENU>GAME")    
        venDLL.command('GAME>GAMEON')
        venDLL.command('GAME>ENDGAME')
        
        interval_series = vensim.get_data(r'C:\workspace\EMA-workbench\src\sandbox\sils\Models\Base\Test.vdf',VOI)
        
#        for key, value in case.items():
#            print key, vensim.get_data(r'C:\workspace\EMA-workbench\src\sandbox\sils\Models\Base\Test.vdf',key)[0], value
        ax1.plot(interval_series, label='new', c='b')
        ax1.plot(behavior, label='original', c='r')
        ax1.legend(loc='best')
    plt.show()