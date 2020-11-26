# Gentrified_Tracts
Using a Multilayer Perceptron (a neural network of relatively low complexity) to predict whether or not a census tract is gentrified.

Explanation of code is fairly thoroughly commented in the coding documents, project is ongoing.
Right now, I've largely been spending time getting to know the problem and datasets thoroughly, 
it helps that I'm actually interested in the problem at hand! As someone living in California,
many cities and communities are subject to the forces of gentrification, and I've seen where
I live, Riverside, visibly gentrify. I'm happy about the diverse people I grew up with thanks to
the fact that Riverside was not an expensive, unobtainable community to live in in the past, but
now that is changing. But how can we best categorize census tracts (which are about the size of
a large neighborhood) as gentrified or not? Use a massive dataset and neural networks of course!

I found the LTDB_DP dataset while looking for census data to use, which is very useful, as the
LTDB_DP dataset has been built to be easily comparable to past census tract data!

Link to LTDB info: https://s4.ad.brown.edu/projects/diversity/Researcher/LTDB.htm

Link to Kaggle set: https://www.kaggle.com/mrmorj/gentrification-and-demographic-analysis

I will use the LTDB_DP dataset most, as it has credibility with the professors who have created
this dataset, while the Kaggle dataset isn't too transparent on where each piece of data comes
from. Therefore, the Kaggle dataset will only be used to give me census tracts of gentrified
communities. This is a potential fault of my design of this project, that I trust the person who
put this Kaggle data together. Hopefully, their choice of 8,000 census tracts which they claimed
to be gentrified, were good choices of tracts. I will take their choice at face value, then
build my neural network around the LTDB_DP dataset with gentrified data in its own category.

Outcome:

After tweaking with the multilayer perceptron (MLP) somewhat, I've built a fairly simple binary
classification model here! The model plateaus at around 80% accuracy, with weighted categories.
Without weighted classes, the model has much higher accuracy, but gives almost no consideration
to the minority case of gentrified tracts. I believe that this network could be greatly improved
by conducting statistical tests, and tests using linear algebra (PCA) to see which variables out
of those I've narrowed the model down to, truly affect the outcome. I also believe that this is
problem which likely can't be modeled by a high accuracy MLP given the design of our project. My
design is fundamentally flawed because I took it for granted that a random Kaggle user knew which
census tracts were gentrified, and which were not, but I have no idea how they decided on their
choices. I believe they chose tracts from cities which are known to be gentrified, but how they
chose specific tracts is a mystery to me.
