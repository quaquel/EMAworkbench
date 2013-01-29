'''
Created on Sep 24, 2012

@author: sibeleker
'''
from mmap import mmap,ACCESS_READ
from xlrd import open_workbook
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

book = open_workbook('H:\My Documents\TF_SA_Article\double_output.xlsx')
sheet = book.sheet_by_index(0)
results = []
print sheet.nrows
#for j in range(sheet.nrows-1):
#    print sheet.cell_value(j+1, 0)
#print sheet.cell_value(1, 0), sheet.cell_value(32,3)
names = []
values = []
for i in range(sheet.ncols):
    names.append(sheet.cell_value(0,i))
    values.append([sheet.cell_value(j+1, i) for j in range(sheet.nrows-1)])
#    results.append((name, values)) 

#names, values1 = results
matplotlib.rcParams['axes.unicode_minus'] = False
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(values[names.index('p2')], values[names.index('S')], 'o')
ax.set_title('S vs p2')
plt.show()