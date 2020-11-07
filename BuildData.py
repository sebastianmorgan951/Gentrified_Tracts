import pandas as pd
import numpy as np
from sklearn import preprocessing

NOCHANGE_COLUMNS = ['l_total','l_nhw','l_under18','l_age25up','l_coll4yr','l_medhhinc','l_occunit',
    'l_ownerocc','l_renterocc','l_pctnhw','l_pctunder18','l_pctcoll4yr','l_pctowner']
CHANGE_COLUMNS = ['total','nhw','under18','age25up','coll4yr','medhhinc','occunit',
    'ownerocc','renterocc','pctnhw','pctunder18','pctcoll4yr','pctowner']
NON_HELPFUL_COLUMNS = ['trtid10','trtid10num','cntyid10','changetype_1pct','changetype_5pct',
    'newtype_1pct','newtype_5pct']

    #These are columns from the LTDB dataset. The NOCHANGE_COLUMNS list contains the names of columns
    #of data which are most accurate at describing census tracts that haven't had boundary changes over time
    #The CHANGE_COLUMNS are more accurate at describing census tracts that have had boundary changes.
    #The NON_HELPFUL_COLUMNS are simply columns that are used to identify tracts, they won't contribute
    #anything to the neural network

df = pd.read_csv('LTDB_DP.csv')
df_flawed_gentrified = pd.read_csv('census_tracts.csv') ##Read CSV datasets
print("General idea of what the LTDB_DP dataset looks like:")
print(df.describe()) ##Getting an idea of what the main dataset we'll use looks like
print("Do we have any columns with 0 or negative values where they shouldn't have those values?")
print(df.describe().loc['min']) ##Finding the minimum value in each columns
#Saw no 0 values, this is likely due to the fact that any microdata released by the government
#has random noise added to it to protect the individual data it has collected. Therefore, I
#don't expect any 0 values, but we do see negative values which surprisingly isn't a problem!
#This is because the data from the LTDB dataset has random noise added, but that noise comes
#from a symmetric distribution with a mean of 0, therefore over large amounts of data, this
#noise does not change any of the overall characteristics of the dataset. We could make the
#negative values into values of 0 instead so they would make more sense to us, but this would
#mean that we would be changing the noise function, so our data and statistics on our data would
#shift upwards, and the only benefit would be that the new non-negative data would look better to us
#Weighing the costs and benefits, it seems that allowing negative values produced by random noise
#is the best choice to preserve general trends within the dataset
print("====================================================")

#Going back to the source of the dataset, the authors say that when the tract hasn't changed
#boundaries, meaning it has a value of "1" in the changetype_1pct, the l_'variablename' columns
#are more accurate, but in other cases, the 'variablename' columns are instead more accurate
df_list = df.values.tolist() #Convert to list, as iterating through this is less computationally intensive
offset = len(NOCHANGE_COLUMNS) #how far apart the same data, derived in different ways, are stored
for i in range(len(df)): #Iterating through every row in our dataframe
    if df_list[i][3] == 1: #If we're dealing with a census tract that hasn't changed boundaries
        for j, col in enumerate(CHANGE_COLUMNS): #Iterating through all the columns we want to change
            df_list[i][j+7+offset] = df_list[i][j+7] #Go to the list storing the dataframe, replace
                                                    #the less accurate data with the more accurate data
df_total = pd.DataFrame(df_list, columns = df.columns)

print("Do both datasets we're combining store tract numbers as integers?")
print(type(df_total.loc[5,'trtid10num']) == type(df_flawed_gentrified.loc[5,'geoid']))
#Ensuring both datasets store tract numbers as integers
print("====================================================")

df_total['gentrif'] = (df_total.trtid10num.isin(df_flawed_gentrified.geoid.values)).astype(int)
#Here, we know exactly where the LTDB_DP dataset comes from, and that it has been improved and worked upon.
#The Kaggle dataset (stored as df_flawed_gentrified) is really only of use to us in the sense that it
#has a list of census tract values which are said to be gentrified. Other than that, I have no idea where most of
#the data from the Kaggle set came from, so we're only using the Kaggle set to get a list of gentrified tracts.
#Now, we know that the Kaggle set was built largely using the LTDB_DP dataset, so I have no worries that the tract
#numbers used with the Kaggle set won't align with the tract values assigned to our more useful LTDB dataset.
#What we do here is check for tract numbers in both the Kaggle and LTDB dataset, and when we find a tract number that
#appears in both, it gets assigned a TRUE value (converted to 1), and other tract numbers get assigned a 0, and
#all of this appears in a new colum in the dataframe, built onto the original LTDB_DP dataframe, called 'gentrif'

#TLDR: Census tract numbers thought to be gentrified get a value of 1, else they get a 0 in a new column of our
#dataframe, called 'gentrif'

df_total = df_total.drop(NOCHANGE_COLUMNS,axis=1)
#Here, we drop all of the columns stored in BAD_COLUMNS from our dataset, as we know they are simply less
#accurate versions of the following columns in the dataset

print("Do we have any null values in the matrix?")
print(df_total.isnull().values.any())
print("====================================================")

usefulCols = df_total.columns[7:30] #These are the numerical columns we can use for our Neural Network
print("Show us the types of variables stored in each column of our df. Any non-numerical columns?")
print(df_total.dtypes) #Even though each of these columns should store numbers, we need to check if they are
#truly stored as ints/floats rather than strings/objects
print("====================================================")
#Here, we see that we do indeed have a column, 'medhhinc', which is stored as objects rather than ints or floats

df_total['medhhinc'] = pd.to_numeric(df_total.medhhinc, errors='coerce')
#Make the medhhinc column type into a column of floats
#By using the errors='coerce' parameter, we may turn objects which aren't easy to convert into NaN values
#Therefore, we can check if this is the case after we have converted this column by running the function we already used
print("Do we have any null values in the matrix after some of these tweaks?")
print(df_total.isnull().values.any())
print("====================================================")

print("Null value in the medhhinc column?")
#We know the null value would be in the medhhinc column, as this is the only column we've changed since we last
#checked for null values
print(df_total.medhhinc.isnull().values.any())
print("How many null values in this medhhinc column?")
print(df_total.medhhinc.isnull().sum())
print("====================================================")
#We find 131 null values in a column with 71,628 entries, so this isn't too bad.
#In fact, since we know medhhinc stands for median household income, which is a continuous variable, we could
#replace these NaN values with the average of the entire column, but since we have enough data, dropping 162
#null rows shouldn't be too big of a deal, and helps make sure we alter the data as little as possible.

df_total = df_total.dropna()
print("Did we really only drop 131 rows, as we want to?")
print(71628-131 == len(df_total))
print("====================================================")

df_total = df_total.drop(NON_HELPFUL_COLUMNS,axis=1) #Drop columns that won't contribute to our neural network

columns = df_total.columns
scaler = preprocessing.StandardScaler()
#Creating a scaler object, this will help us standardize each column of our dataframe with a mean = 0
#and a variance = 1. We need to create this "scaler" object, as our dataframe has widely varying columns,
#so this object better deals with such situations than other options this sklearn library provides us with
df_total_scaled = scaler.fit_transform(df_total)  #Give each data column a mean of 0 and variance of 1 so the
#distribution of data in each column doesn't inadvertently give any one variable more weight than another
df_total_scaled = pd.DataFrame(df_total_scaled, columns=columns) #convert back into Pandas DataFrame
df_total_scaled['gentrif'] = df_total['gentrif'] #We don't want to scale the row we are predicting, however
print("After scaling our data, how are our variables distributed?")
print(df_total_scaled.describe().loc[['mean','std','max'],].round(2).abs())
print("====================================================")

print("Storing processed data as a csv file called 'ProcessedData.csv'")
df_total_scaled.to_csv('ProcessedData.csv', index=False)
#Save these dataframes for easy future use
