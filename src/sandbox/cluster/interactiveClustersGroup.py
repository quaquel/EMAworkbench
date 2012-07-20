#!/usr/bin/env python
"""
Draws some x-y line and scatter plots. On the left hand plot:
 - Left-drag pans the plot.
 - Mousewheel up and down zooms the plot in and out.
 - Pressing "z" brings up the Zoom Box, and you can click-drag a rectangular 
   region to zoom.  If you use a sequence of zoom boxes, pressing alt-left-arrow
   and alt-right-arrow moves you forwards and backwards through the "zoom 
   history".
"""

#Import output from cPickle and prepare data
import cPickle
import operator
import numpy as np

# Enthought library imports
from enthought.enable.api import BaseTool
from enthought.traits.api import List, Instance, Any, HasTraits, Str, Float
from enthought.traits.ui.api import Item, Group, View, TabularEditor
    
from enthought.traits.ui.tabular_adapter import TabularAdapter

# Chaco imports
from enthought.chaco.api import ArrayPlotData, Plot, GridContainer
from enthought.chaco.tools.api import PanTool, ZoomTool 


# Enthought library imports
from enthought.enable.api import Component, ComponentEditor
colors = ['blue', 'red', 'green']

class CaseAdapter ( TabularAdapter ):
    columns = [ ( 'Uncertainty', 'name' ), ( 'Value', 'value' ) ]

class UncertaintyValue ( HasTraits ):
    name   = Str
    value = Str
        
#==============================================================================
# # Demo class that is used by the demo.py application.
#==============================================================================
class Demo(HasTraits):
    plot = Instance(Component)
    fileName = "clusters.cpickle"
    case = List( UncertaintyValue )
    
    cases = {}
    defaultCase = []

    # Attributes to use for the plot view.
    size=(400,1600)
    
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
    
    def setFileName(self, newName):
        self.fileName = newName
    
    def _update_case(self, name):
        
        if name:
            self.case = self.cases.get(name)
            
        else:
            self.case = self.defaultCase
            
    
    def _plot_default(self):
        
        #load the data to visualize. 
        # it is a list of data in the 'results' format, each belonging to a cluster - gonenc
        resultsList = cPickle.load(open(self.fileName, 'r'))
        
        
        #get the names of the outcomes to display
        outcome = []
        for entry in resultsList:
            a = entry[0][1].keys()
            outcome.append(a[0])
        
#        outcome = resultsList[0][0][1].keys()
        
        # pop the time axis from the list of outcomes
#        outcome.pop(outcome.index('TIME'))
        x = resultsList[0][0][1]['TIME']
        
        # the list and number of features (clustering related) stored regarding each run
        features = resultsList[0][0][0][0].keys()
        noFeatures = len(features)
    
        # Iterate over each cluster to prepare the cases corresponding to indivisdual runs in
        # each cluster plot. Each case is labeled as, e.g., y1-2 (3rd run in the 2nd cluster) - gonenc
        for c, results in enumerate(resultsList):
            for j, aCase in enumerate(results):
                aCase = [UncertaintyValue(name=key, value=value) for key, value in aCase[0][0].items()]
                self.cases['y'+str(c)+'-'+str(j)] = aCase
        
#        for j, aCase in enumerate(results):
#            aCase = [UncertaintyValue(name="blaat", value=aCase[0][0])] 
#            self.cases['y'+str(j)] = aCase
        
        #make an empty case for default. 
        #if you have multiple datafields associated with a run, iterate over
        #the keys of a dictionary of a case, instead of over lenght(2)
        case = []
        for i in range(noFeatures):
            case.append(UncertaintyValue( name = 'Default',value = 'None'))
        self.case = case
        self.defaultCase = case
    
    
        # Create some x-y data series to plot
        pds = []
        # enumerate over the results of all clusters    
        for c, results in enumerate(resultsList):
            pd  = ArrayPlotData(index = x)
            for j in range(len(results)): 
                data = np.array(results[j][1].get(outcome[c]))
                print "y"+str(c)+'-'+str(j)
                pd.set_data("y"+str(c)+'-'+str(j),  data)
            pds.append(pd)
        
        # Create a container and add our plots
        container = GridContainer(
                                  bgcolor="lightgray", use_backbuffer=True,
                                  shape=(len(resultsList), 1
                                         ))
    
        #plot data
        tools = []
        for c, results in enumerate(resultsList):
            pd1 = pds[c]
    
            # Create some line plots of some of the data
            plot = Plot(pd1, title='Cluster '+ str(c), border_visible=True, 
                        border_width = 1)
            plot.legend.visible = False
        
            #plot the results
            for i in range(len(results)):
                plotvalue = "y"+str(c)+'-'+str(i)
                print plotvalue
                color = colors[i%len(colors)]
                plot.plot(("index", plotvalue), name=plotvalue, color=color)
                
            #make sure that the time axis runs in the right direction
            for value in plot.plots.values():
                for entry in value:
                    entry.index.sort_order = 'ascending'
    
            # Attach the selector tools to the plot
            selectorTool1 = LineSelectorTool(component=plot)
            plot.tools.append(selectorTool1)
            tools.append(selectorTool1)
            
            # Attach some tools to the plot
            plot.tools.append(PanTool(plot))
            zoom = ZoomTool(component=plot, tool_mode="box", always_on=False)
            plot.overlays.append(zoom)
        
            container.add(plot)
    
        #make sure the selector tools knows the main screen
        for tool in tools:
            tool._demo = self
    
        return container



class LineSelectorTool(BaseTool):
    """ LineSelectorTool 
    """
    
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
        
#        for selector in self._other_selectors:
#            selector.updateSelection(name)
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
 
def plotForMeSam ():
    demo = Demo()
    demo.configureTraits()

if __name__ == "__main__":
    demo = Demo()   
    demo.configure_traits()