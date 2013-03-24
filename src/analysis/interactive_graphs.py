'''
Created on 1 august. 2011

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
.. codeauthor:: chamarat<c.hamarat(at) tudelft (dot) nl>

This module offers functionality for creating interactive graphs. There are 
currently two types of interactive graphs: a line graph and a scatter plot.

In both cases, you can select a run and see the experiment that generated that
run. The selected run is highlighted, and if there are other outcomes of 
interests, the results for the selected run are also highlighted in those
graphs. 


'''

import operator
import numpy as np

# Enthought library imports
from enthought.enable.api import BaseTool
from enthought.traits.api import List, Instance, Any, HasTraits, Str, String, Range, Int, Float
from enthought.traits.ui.api import Item, Group, View, TabularEditor
from enthought.traits.ui.tabular_adapter import TabularAdapter

# Chaco imports
from enthought.chaco.api import ArrayPlotData, Plot
from enthought.chaco.tools.api import PanTool, ZoomTool 
from enthought.chaco.api import  GridContainer

# Enthought library imports
from enthought.enable.api import Component, ComponentEditor
import expWorkbench.util as util
colors = ['blue', 'red', 'green']


__all__ = ['make_interactive_plot']

def make_interactive_plot(results, outcomes = [], type='lines'):
    '''
    Function that makes an interactive plot. The interactive plots assume
    time series data. 
    
    :param results: the results as returned by :meth:`perform_experiments`
    :param type: type of plot. Currently there is an interactive line
                 plot and an interactive scatter plot. Possible values are
                 'lines' and 'scatter'.
    
    '''    
    if type == 'lines':
        demo = make_lines(results)
    elif type == 'scatter':
        demo = InteractiveMultiplot(results)
    demo.configure_traits()    



    

#==============================================================================
# # helper classes for showing a case
#==============================================================================
class CaseAdapter ( TabularAdapter ):
    columns = [ ( 'Uncertainty', 'name' ), ( 'Value', 'value' ) ]

class UncertaintyValue ( HasTraits ):
    name   = Str
    value = Str


#==============================================================================
# interactive screens
#==============================================================================  

class AbstractScreen(HasTraits):
    '''
    Base class from which interactive screens can be derived.
    
    '''
    
    def _prepare_data(self):
        '''
        helper method that prepares the data. In assumes that 
        self.input contains the tuple returned by :meth:`perform_experiments`.
        It uses this to make a list containing the experiments. Where each
        experiment is a list in turn. The outcomes are checked to see
        if they contain a time dimension.  
        
        
        '''
        
        cases, results = self.input
            
        #establish time axis
        try:
            time =  results.pop('TIME')[0,:]
        except KeyError:
            time =  np.arange(0, results.values()[0].shape[1])[np.newaxis, :]
            
        #establish outcomes to plot
        outcomes = results.keys()
        outcomes.sort()
    
        #build cases
        uncertainties =  []
        for entry in cases.dtype.descr:
            uncertainties.append(entry[0])
    
        for i in range(cases.shape[0]):
            aCase = [UncertaintyValue(name=uncertainty, value=str(cases[uncertainty][i])) 
                     for uncertainty in uncertainties]
            self.cases['y'+str(i)] = aCase
      
        #creating an empty case to use as default      
        case = []
        for uncertainty in uncertainties:
            case.append(UncertaintyValue( name = uncertainty,value = str(0)))
        self.case = case
        self.defaultCase = case
    
        
        return outcomes, results, time




def make_lines(results):
    '''
    helper function for making an interactve lines plot
    
    :param results: the return from :meth:`perform_experiments`.
    :return: an :class:`InteractiveLines` instance.
    
    '''
    
    results = results
    
    
    class InteractiveLines(AbstractScreen):
        '''
        An interactive lines screen
        
        '''
        
        size=(600, 200*results[0].shape[0])
        
        #traits
        plot = Instance(Component)
        case = List( UncertaintyValue )
        
        #specification of the view
        traits_view = View(Group(
                        Group(
                            Item('plot', 
                                 editor=ComponentEditor(size=size), 
                                 show_label=False),
                            orientation = "vertical",
                            show_border=True,
                            scrollable=True
                            ),
                        Group(
                            Item( 'case', 
                                  editor = TabularEditor(adapter = CaseAdapter(
                                                                can_edit=False)),
                                  show_label = False),
                            orientation = "vertical",
                            show_border=True
                            ),
                        layout = 'split',
                        orientation = 'horizontal'),
                      title     = 'Interactive Lines', 
                      resizable = True
                      )
        
        def __init__(self, results):
            super(InteractiveLines, self).__init__()
            
            #other attributes
            self.cases = {}
            self.defaultCase = []
            self.input = results
        
        def _update_case(self, name):
            if name:
                self.case = self.cases.get(name)
            else:
                self.case = self.defaultCase
             
        def _plot_default(self):
            outcomes, results, time = self._prepare_data()
            
            # get the x,y data to plot
            pds = []    
            for outcome in outcomes:
                pd  = ArrayPlotData(index = time)
                result = results.get(outcome)
                for j in range(result.shape[0]): 
                    pd.set_data("y"+str(j), result[j, :] )
                pds.append(pd)
            
            # Create a container and add our plots
            container = GridContainer(
                                      bgcolor="white", use_backbuffer=True,
                                      shape=(len(outcomes),1))
        
            #plot data
            tools = []
            for j, outcome in enumerate(outcomes):
                pd1 = pds[j]
        
                # Create some line plots of some of the data
                plot = Plot(pd1, title=outcome, border_visible=True, 
                            border_width = 1)
                plot.legend.visible = False
            
                a = len(pd1.arrays)- 1
                if a > 1000:
                    a = 1000
                for i in range(a):
                    plotvalue = "y"+str(i)
                    color = colors[i%len(colors)]
                    plot.plot(("index", plotvalue), name=plotvalue, color=color)
                    
                for value in plot.plots.values():
                    for entry in value:
                        entry.index.sort_order = 'ascending'
        
                # Attach the selector tools to the plot
                selectorTool1 = LineSelectorTool(component=plot)
                plot.tools.append(selectorTool1)
                tools.append(selectorTool1)
                container.add(plot)
        
            #make sure the selector tools know each other
            
            for tool in tools:
                a = set(tools) - set([tool])
                tool._other_selectors = list(a)
                tool._demo = self
        
            return container
    
    demo = InteractiveLines(results)
    return demo
    
          
class InteractiveMultiplot(AbstractScreen):
    '''
    An interactive scatter multiplot screen.
    
    '''    
    
    #traits
    low = Int(0)
    high = Int(1)
    
    startValue = 'low'
    
    range = Range(low='low', high='high', value=startValue)
    plot = Instance(Component)
    case = List(UncertaintyValue)

    size = (900,500)
    
    #layout of screen
    traits_view = View(Group(
                        Group(Item('plot', 
                                   editor=ComponentEditor(size=size), 
                                   show_label=False),
                             Item( '_' ),
                             Item('range',
                                  show_label=False)),     
                        Group(Item('case', 
                                   editor = TabularEditor(adapter=CaseAdapter(can_edit=False)),
                                   show_label=False)),
                    layout = 'split',
                    orientation = 'horizontal'),
                  title     = 'scatter', 
                  resizable = True
                  )
    
    def __init__(self, results):
        super(InteractiveMultiplot, self).__init__()
        #other attributes
        self.cases = {}
        self.defaultCase = []
        self.input = results
        self.data = {}
   
   
    def _update_case(self, name):
        if name != None :
            self.case = self.cases.get('y'+str(name))
        else:
            self.case = self.defaultCase
   
    def _plot_default(self):
        outcomes, results, time = self._prepare_data()
        
        self.outcomes = outcomes
        self.time = time
        self.low = int(np.min(time))
        self.high = int(np.max(time))
        

        self.data = {}
        for outcome in outcomes:
            self.data[outcome] = results[outcome]
        
        # Create some data
        pd = ArrayPlotData()
        
        for entry in outcomes:
            if self.startValue == 'low':
                pd.set_data(entry, self.data[entry][:,0])
            elif self.startValue == 'high':
                pd.set_data(entry, self.data[entry][:,-1])
        self.plotdata = pd
        
#        outcomes = outcomes[0:3]

        container = GridContainer(shape=(len(outcomes),len(outcomes)))
        combis = [(field1, field2) for field1 in outcomes for field2 in outcomes]
        
        selectorTools = []
        for entry1, entry2 in combis:

            # Create the plot
            if entry1 == entry2:
                plot = Plot(pd)
            else:
                plot = Plot(pd)
                
                #set limits for x and y to global limits
                plot.range2d._xrange._high_setting = np.max(self.data[entry1])
                plot.range2d._xrange._high_value = np.max(self.data[entry1])
                plot.range2d._xrange._low_setting = np.min(self.data[entry1])
                plot.range2d._xrange._low_value = np.min(self.data[entry1])
                
                plot.range2d._yrange._high_setting =np.max(self.data[entry2])
                plot.range2d._yrange._high_value = np.max(self.data[entry2])
                plot.range2d._yrange._low_setting = np.min(self.data[entry2])
                plot.range2d._yrange._low_value = np.min(self.data[entry2])

                #make the scatter plot
                plot.plot((entry1, entry2),
                            type="scatter",
                            marker="circle",
                            color="blue",
                            marker_size=3,
                            bgcolor="white")[0]                
            
                tool = ScatterSelectorTool(component=plot)
                tool._index = entry1
                plot.tools.append(tool)
                selectorTools.append(tool)
            
            plot.height = 500
            plot.width = 500
            plot.border_width = 1
            plot.aspect_ratio  = 1.
            
            container.add(plot)
        
        for tool in selectorTools:
            tool._other_selectors = list(set(selectorTools) - set([tool]))
            tool._demo = self
        return container

    def _range_changed(self, old, new):
        #modify the data used in the scatter plots by updating them
        #see data_chooser example in the chaco examples
        
        # Create some data
        a = self.time-new
        a[a < 0 ] = 100
        index = np.argmin(a)
        for entry in self.outcomes:
            self.plotdata.set_data(entry, self.data[entry][:,index])
        
        for entry in self.plot._components:
            if entry.tools:
                point = entry.tools[0]._selected_point
                entry.tools[0].updateSelection(None) #reset the cached selection
                entry.tools[0].updateSelection(point) #set the selection anew


#==============================================================================
# tools 
#==============================================================================
class LineSelectorTool(BaseTool):
    """ LineSelectorTool"""
    
    _selected_line = Any
    _demo = Any
    _other_selectors = List(Any)
  
    def normal_left_down(self, event):
        """ Handles the left mouse button being pressed when the tool is in the
        'normal' state.

        If selecting is enabled and the cursor is within **threshold** of a
        data point, the method calls the subclass's _select" or _deselect
        methods to perform the appropriate action, given the current
        selection_mode.
        """
        
        name = self._get_selection_state(event)
        self.updateSelection(name)
        
        for selector in self._other_selectors:
            selector.updateSelection(name)
        self._demo._update_case(name)
        
        return
    
    def updateSelection(self, name):
        if name:
            selected = self.component.plots.get(name)[0]
            
            if selected != self._selected_line:
                self._select(selected)
                if self._selected_line:
                    self._deselect(self._selected_line)
                self._selected_line = selected
        else:
            if self._selected_line:
                self._deselect(self._selected_line)
                self._selected_line = None

    def _get_selection_state(self, event):
        """ Returns a tuple reflecting the current selection state

        Parameters
        ----------
        event : enable KeyEvent or MouseEvent

        Returns
        -------
        (already_selected, clicked) : tuple of Bool
            clicked is True if the item corresponding to the input event has
            just been clicked.
            already_selected indicates that the item corresponding to the
            input event is already selected.
        
        """
        xform = self.component.get_event_transform(event)
        event.push_transform(xform, caller=self)
       
        renders = []
        for element in self.component.plots.keys():
            for plot in self.component.plots.get(element):
                a = plot.hittest(np.array((event.x, event.y)))
                if a:
                    renders.append((element))
        if renders:
            return renders[0]
        else:
            return None
  
    def _select(self, selected):
        """ Decorates a plot to indicate it is selected """
        for plot in reduce(operator.add, selected.container.plots.values()):
            if plot != selected:
                plot.alpha /= 3
            else:
                plot.line_width *= 2

        plot.request_redraw()

    def _deselect(self, selected):
        for plot in reduce(operator.add, selected.container.plots.values()):
            if plot != selected:
                plot.alpha *= 3
            else:
                plot.line_width /= 2

        plot.request_redraw()


class ScatterSelectorTool(BaseTool):
    """ 
    ScatterSelectorTool 
    """
    
    _selected_case = Any
    _demo = Any
    _other_selectors = List(Any)
    
    _threshold = 5.0
    _index = String
    _selected_point = None
    _demo = Any
  
    def normal_left_down(self, event):
        """ Handles the left mouse button being pressed when the tool is in the
        'normal' state.

        If selecting is enabled and the cursor is within **threshold** of a
        data point, the method calls the subclass's _select" or _deselect
        methods to perform the appropriate action, given the current
        selection_mode.
        """
        
        index = self._get_selection_state(event)
        self.updateSelection(index)
        
        for tool in self._other_selectors:
            tool.updateSelection(index)
        self._demo._update_case(index)
        
        return
    
    def updateSelection(self, index):
        if index != None:
            self._select(index)
            self._selected_point = index
        else:
            self._deselect()
            self._selected_point = None
        

    def _get_selection_state(self, event):
        """ Returns a tuple reflecting the current selection state

        Parameters
        ----------
        event : enable KeyEvent or MouseEvent

        Returns
        -------
        index : the index of the selected point, or None in case none is selected
        
        """
        xform = self.component.get_event_transform(event)
        event.push_transform(xform, caller=self)
        index = self.component.plots['plot0'][0].map_index((event.x, event.y), 
                                                         threshold=self._threshold)

        return index
  
    def _select(self, index):
        #this works, appart from the 'index' key, this name should be
        #based on the fields being plotted
        datasource = self.component.datasources.get(self._index)
        datasource.metadata['selections'] = [index]
        
    def _deselect(self):
        #this works, appart from the 'index' key, this name should be
        #based on the fields being plotted
        datasource = self.component.datasources.get(self._index)
        datasource.metadata['selections'] = []

#==============================================================================
# main
#==============================================================================
if __name__ == "__main__":
    file = r'chacotest.cpickle'
    results = util.transform_old_cPickle_to_new_cPickle(file)
    make_interactive_plot(results, type='lines')
