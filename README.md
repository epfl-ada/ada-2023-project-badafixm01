# TITLE

*Berta Céspedes, Inès Kahlaoui, Mya Lahjouji, Fernando Meireles, Xiaocheng Zhang*

## Abstract
How can a movie genre achieve success in box office revenue? We primarily seek the answer in the realm of emotions. Film enthusiasts and casual viewers alike often base their movie choices on emotions they wish to experience, whether it's thrill, romance, fear, or joy. Utilising the movie plot summaries and their corresponding metadata, we could use NLP techniques to extract sentiment scores from these summaries. Our hypothesis is that certain sentiments, like joy or suspense, may consistently align with higher box office revenues for some genres. By correlating sentiment scores with box office revenues for different genres and potentially factoring in character gender, release dates, countries and cultural contexts, this analysis aims to reveal the underlying emotional 'décor' that drives moviegoers to the cinema. This investigation could not only provide valuable insights for film producers but also paint a picture of society's emotional needs and preferences at different times, across different countries.

## Research Question
* Which emotions are more common in which movie genres? What are the typical patterns of emotion intensities for the most prevalent genres?
* Are emotion intensities and sentiment score good predictors for box office revenue? How do other factors such as gender, runtime, country, etc. come into play?
* How do movie sentiments compare over different time periods and cultural context?

## Methods
### Preprocessing
We apply a series of transformation to the movie summaries, including tokenization, lemmatization, removing punctuations, and converting to lowercase. These preprocessing steps ensures proper recognition of words associated with emotions that are defined in the emotion lexicon.

### Genre-Emotion Analysis
This analysis is conducted on a per genre basis, focusing on the top-10 prevalent genres while considering the first genre associated with each genre as the main genre.

#### Emotion Quantification 
We employ two metrics for evaluating emotions in movie plot summaries and titles.

**Emotion intensity vector:** This metric is derived from the [NRC Emotion Intensity Lexicon](https://saifmohammad.com/WebPages/AffectIntensity.htm), which categorizes English words into 8 emotion categories and assigns an emotion intensity score to each word. We compute the emotion intensity vector for each summary by aggregating the intensities of each lexicon word and normalization, such that the entries in the vector indicate relative strength of the 8 emotions.

**Sentiment score:** We use the [TextBlob](https://textblob.readthedocs.io/en/dev/) library for calculating sentiment scores for movie summaries and titles. The sentiment scores are further classified into 5 categories, which are used as labels in machine learning.

#### Cluster Analysis
To identify the prevalent emotions and typical emotion intensity patterns for each genre, we perform cluster analysis on emotion intensity vectors. This analysis answers our first research question.

#### Genre Definition by Emotion
We extend the idea of linking genres with emotions by creating a definition of each genre with emotional tags. For this analysis we utilize the [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) pipeline, which contains 27 emotion categories and provides a pretrained BERT model for inferencing.

#### Time-Based Emotional Trends
As people's tastes evolve, the preferred emotion might shift over the years. For example, in an era dominated by socio-political unrest, audiences might lean towards uplifting films as an escapade. We exploit this idea by two approaches: 
**Time-Series Analysis:** Develop line plots for genre-specific emotion intensities over time. 
**Cohort analysis:** Segment the data into cohorts based on movie release years or periods and compare emotion intensity trends across cohorts.

### Emotion as a Predictor
#### Machine Learning
We adopt machine learning techniques such as decision tree and random forest for predicting missing box office revenue data. We start with genre and sentiment score as the only input features, and iteratively add to the complexity of the model by considering other variables, e.g., movie runtime, character gender, language, and country. The aim is to find a subset of variables that allow us to determine the success of a movie.

#### Regression Analysis
We conduct multivariate regression per genre on box office revenue with emotion intensities as independent variables. With normalized intensity vectors, the regression coefficients can indicate the relative importance of emotions to movie revenue success for each genre. 

## Proposed Timeline

## Organization within the Team