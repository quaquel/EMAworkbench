import matplotlib 
import matplotlib.pyplot as plt


im=matplotlib.image.imread(r'./data/adaptive vs masterplan regret masterPlan better.png')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1], 
                  aspect='auto', 
                  frameon=False, 
                  xticks=[], 
                  yticks=[])

ax.imshow(im)
plt.savefig("environment_and_planning_b.png", dpi=75)
