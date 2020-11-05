import matplotlib
matplotlib.use("TkAgg")
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('ProcessedData.csv')

usefulCols = df.columns[7:30]

df.hist()
plt.show()

plt.subplots(5,5,figsize=(20,20))

# Plot a density plot for each variable
for i, col in enumerate(usefulCols):
    ax = plt.subplot(5,5,i+1)
    ax.yaxis.set_ticklabels([])
    sns.distplot(df.loc[df.gentrif == 0][col], hist=False, axlabel= False, kde_kws={'linestyle':'-', 'color':'black', 'label':"Not Gentrified"})
    sns.distplot(df.loc[df.gentrif == 1][col], hist=False, axlabel= False, kde_kws={'linestyle':'--', 'color':'black', 'label':"Gentrified"})
    ax.set_title(col)

# Hide the 9th subplot (bottom right) since there are only 8 plots
plt.tight_layout()
plt.show()
