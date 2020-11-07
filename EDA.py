import matplotlib
matplotlib.use("TkAgg")
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('ProcessedData.csv') #Read our processed, scaled data

df.hist()  #Give us a quick view of the distribution of all of our variables
plt.show()

plt.subplots(5,5,figsize=(20,20))
#Initialize an empty 5x5 grid of plots

# Plot a density plot for each variable
for i, col in enumerate(df.columns): #Iterate through each column
    ax = plt.subplot(5,5,i+1) #For each column, start graphing on a new subplot
    ax.yaxis.set_ticklabels([]) #No y axis markers, they provide unnecessary info
    sns.distplot(df.loc[df.gentrif == 0][col], hist=False, axlabel= False,
        kde_kws={'linestyle':'-', 'color':'black', 'label':"Not Gentrified"})
        #For the non-gentrified data in the column we're looking at, plot that
        #column's distribution as a solid line
    sns.distplot(df.loc[df.gentrif == 1][col], hist=False, axlabel= False,
        kde_kws={'linestyle':'--', 'color':'black', 'label':"Gentrified"})
        #On the same subplot as above, plot the column's distribution for
        #gentrified data in a dashed line
    ax.set_title(col)
    #give each subplot a title so we know what variable we're looking at

plt.tight_layout()
plt.show()
