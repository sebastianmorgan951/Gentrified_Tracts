import matplotlib
matplotlib.use("TkAgg")
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('Datasets/BuildData_Set.csv') #Read our processed, scaled data

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
plt.show() #Show us the all the plots produced side by side

#After seeing these plots, it seems clear that there are a few variables where
#distributions between gentrified and non-gentrified populations are too similar
#to provide the neural network with much information. There are also a few
#variables which are simply repeats of other variables (once we've normalized
#all of the columns), like the "het_nhw" and "hetn_nhw" columns.

#Also, the "cntychng" and "pcntychng" columns are useful, however "pcntychng"
#shows population change proportional to initial population, while "cntychng"
#shows total population change, which provides somewhat less information, so
#we'll only keep the "pcntychng" variable

#Remove all of these unnecessary columns
NonUsefulCols = ["total","under18","occunit","pctunder18","cntychng",
    "hetn_under18","hetn_nhw","hetn_owner","hetn_coll4yr"]
df = df.drop(NonUsefulCols,axis=1) #Drop the less useful columns

print("Storing processed data as a csv file called 'ProcessedData.csv'")
df.to_csv('Datasets/ProcessedData.csv',index=False) #Store our changes
