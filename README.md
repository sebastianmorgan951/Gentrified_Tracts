# Gentrified_Tracts
Using a Multilayer Perceptron (a neural network of relatively low complexity) to predict whether or not a census tract is gentrified.

Explanation of code is fairly thoroughly commented in the coding documents, project is ongoing.
Right now, I've largely been spending time getting to know the problem and datasets thoroughly, 
it helps that I'm actually interested in the problem at hand! As someone living in California,
tons of cities and communities are subject to the forces of gentrification, and I've seen where
I live, Riverside, visibly gentrify. I'm happy about the diverse people I grew up with thanks to
the fact that Riverside was not an expensive, unobtainable community to live in in the past, but
now that is changing. But how can we best categorize census tracts (which are about the size of
a large neighborhood) as gentrified or not? Use a massive dataset and neural networks of course!

I found the LTDB_DP dataset while looking for census data to use, and it is even more useful in
that the LTDB_DP dataset has been built to be easily comparable to past census tract data!

Link to LTDB info: https://s4.ad.brown.edu/projects/diversity/Researcher/LTDB.htm
Link to Kaggle set: https://www.kaggle.com/mrmorj/gentrification-and-demographic-analysis

I will use the LTDB_DP dataset most, as it has credibility with the professors who have created
this dataset, while the Kaggle dataset isn't too transparent on where each piece of data comes
from. Therefore, the Kaggle dataset will only be used to give me census tracts of gentrified
communities. This is a potential fault of my design of this project, that I trust the person who
put this Kaggle data together. Hopefully, their choice of 8,000 census tracts which they claimed
to be gentrified, were good choices of tracts. I will take their choice at face value, then
build my neural network around the LTDB_DP dataset with gentrified data in its own category.
