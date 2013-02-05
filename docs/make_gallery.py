#
# generate a thumbnail gallery of examples
# derived from networkx and matplotlib code


template = """\
{%% extends "layout.html" %%}
{%% set title = "Gallery" %%}


{%% block body %%}

<h3>Click on any image to see source code</h3>
<br/>

%s
{%% endblock %%}
"""
link_template = """\
<a href="%s"><img src="%s" border="0" alt="%s"/></a>
"""

import os, glob, re, shutil, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot
import matplotlib.image
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

sys.path.append(os.path.abspath('../src'))
sys.path.append(os.path.abspath('../src/analysis'))

examples_source_dir = '.\source\gallery'
examples_dir = 'gallery'
template_dir = 'source\ytemplates'
static_dir_thumb = r'build\html\_static'
static_dir_fig = r'build\html\_images'
pwd=os.getcwd()
rows = []

os.chdir(examples_source_dir)
all_examples=sorted(glob.glob("*.py"))

# check for out of date examples
stale_examples=[]
for example in all_examples:
    png=example.replace('py','png')                             
    png_static=os.path.join(pwd, static_dir_fig, png)
    
#    print png_static, example
#    print os.stat(png_static).st_mtime, os.stat(example).st_mtime
#    print (not os.path.exists(png_static))
#    print os.stat(png_static).st_mtime < os.stat(example).st_mtime
    if (not os.path.exists(png_static) or 
        os.stat(png_static).st_mtime < os.stat(example).st_mtime):
        stale_examples.append(example)

for example in stale_examples:
    print example,
    png=example.replace('py','png')
                                 
    matplotlib.pyplot.figure(figsize=(6,6))
    stdout=sys.stdout
    sys.stdout=open('nul','w')
    try:
        execfile(example)
        sys.stdout=stdout
        print " OK"
    except ImportError,strerr:
        sys.stdout=stdout
        sys.stdout.write(" FAIL: %s\n"%strerr)
        continue
    matplotlib.pyplot.clf()
    im=matplotlib.image.imread("./pictures/"+png)
    
    fig = Figure(figsize=(2.5, 2.5))
    canvas = FigureCanvas(fig)
    ax = fig.add_axes([0,0,1,1], 
                      aspect='auto', 
                      frameon=False, 
                      xticks=[], 
                      yticks=[])


    ax.imshow(im, aspect='auto', resample=True, interpolation='bilinear')
    thumbfile=png.replace(".png","_thumb.png")
    fig.savefig("./pictures/"+thumbfile, dpi=75)

#    print os.path.join(pwd,static_dir_thumb,thumbfile) 
    shutil.copy("./pictures/"+thumbfile,os.path.join(pwd,static_dir_thumb,thumbfile))
#    shutil.copy(png,os.path.join(pwd,static_dir,png))

for example in all_examples:
    png=example.replace('py','png') 
    thumbfile=png.replace(".png","_thumb.png")
    
    basename, ext = os.path.splitext(example)
    link = '%s/rst/%s.html'%(examples_dir, basename)
#    print link
    
    loc = os.path.join('_static',thumbfile)
    loc = loc.replace('\\', '/')
    rows.append(link_template%(link, loc, basename))

os.chdir(pwd)
fh = open(os.path.join(template_dir,'gallery.html'), 'w')
fh.write(template%'\n'.join(rows))
fh.close()

#from make_examples_rst import main
#main(examples_source_dir, examples_source_dir)

