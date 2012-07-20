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
from enthought.chaco.api import ArrayPlotData, Plot
from enthought.chaco.tools.api import PanTool, ZoomTool 
from enthought.chaco.api import  GridContainer

# Enthought library imports
from enthought.enable.api import Component, ComponentEditor
colors = ['blue', 'red', 'green']


class CaseAdapter ( TabularAdapter ):
    columns = [ ( 'Things', 'name' ), ( 'Value', 'value' ) ]

class UncertaintyValue ( HasTraits ):
    name   = Str
    value = Str #Float
        
#==============================================================================
# # Demo class that is used by the demo.py application.
#==============================================================================
class Demo(HasTraits):
    plot = Instance(Component)
    fileName = "default.txt"
    case = List( UncertaintyValue )
    cases = {}
    
    defaultCase = []

    # Attributes to use for the plot view.
    size=(400,250)
    
    traits_view = View(Group(
                        Group(
                            Item('plot', 
                                 editor=ComponentEditor(size=size), 
                                 show_label=False),
                            orientation = "vertical",
                            show_border=True
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
        results = cPickle.load(open(self.fileName, 'r'))
        outcomes = results[0][1].keys()
        outcomes.pop(outcomes.index('TIME'))
        x = results[0][1]['TIME']
    
        for j, aCase in enumerate(results):
            aCase = [UncertaintyValue(name=key, value=value) for key, value in aCase[0][0].items()]
            self.cases['y'+str(j)] = aCase
    
        uncertainties = results[0][0][0]
        uncertaintynames = uncertainties.keys()
        uncertaintyvalues = []
        for key in uncertainties.keys():
            uncertaintyvalues.append(uncertainties[key])
        
        case = []
        for i in range(len(uncertainties)):
            case.append(UncertaintyValue( name = str(uncertaintynames[i]),value = "")) #haydaa
        self.case = case
        self.defaultCase = case
    
        # Create some x-y data series to plot
        pds = []    
        for i, outcome in enumerate(outcomes):
            pd  = ArrayPlotData(index = x)
            for j in range(len(results)): 
                pd.set_data("y"+str(j), results[j][1].get(outcome) )
            pds.append(pd)
        
        # Create a container and add our plots
        container = GridContainer(
                                  bgcolor="lightgray", use_backbuffer=True,
                                  shape=(1, 1))
    
        #plot data
        tools = []
        for j in range(len(outcomes)):
            pd1 = pds[j]
    
            # Create some line plots of some of the data
            plot = Plot(pd1, title=outcomes[j], border_visible=True, 
                        border_width = 1)
            plot.legend.visible = False
        
            for i in range(len(results)):
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
            
            # Attach some tools to the plot
            plot.tools.append(PanTool(plot))
            zoom = ZoomTool(component=plot, tool_mode="box", always_on=False)
            plot.overlays.append(zoom)
        
            container.add(plot)
    
        #make sure the selector tools know each other
        
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