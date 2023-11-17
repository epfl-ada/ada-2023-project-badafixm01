# Feel the Genres

*Berta Céspedes, Inès Kahlaoui, Mya Lahjouji, Fernando Meireles, Xiaocheng Zhang*

## Abstract
How can a movie genre achieve success in box office revenue? We primarily seek the answer in the realm of emotions. Film enthusiasts and casual viewers alike often base their movie choices on emotions they wish to experience, whether it is thrill, romance, fear, or joy. Utilising the movie plot summaries and their corresponding metadata, we could use NLP techniques to extract sentiment scores from these summaries. Our hypothesis is that certain sentiments, like joy or suspense, may consistently align with higher box office revenues for some genres. By correlating sentiment scores with box office revenues for different genres and potentially factoring in main character gender, release dates, countries and cultural contexts, this analysis aims to reveal the underlying emotional 'décor' that drives moviegoers to the cinema. This investigation could not only provide valuable insights for film producers but also paint a picture of society's emotional needs and preferences at different times, across different countries.

## Research Question
* Which emotions are more common in which movie genres? What are the typical patterns of emotion intensities for the most prevalent genres?
* Are emotion intensities and sentiment score good predictors for box office revenue? How do other factors such as main character gender, runtime, country, etc. come into play?
* How do movie sentiments compare over different time periods and cultural context?

## Dataset
### [CMU Movie Summary Corpus](https://www.cs.cmu.edu/~ark/personas/)
This is the primary dataset we use in our analysis. Our research questions mostly concern the movie summary, genre, and box office revenue data. During the initial filtering of the dataset, we found 42,207 movies with available summary and metadata, out of which 7,588 movies additionally have box office revenue data. Since the dataset contains a total of 364 genres and some of them may not be meaningful, we only consider the first genre associated with each movie as the main genre and focus on the top-10 frequent main genres.

### [NRC Emotion Intensity Lexicon](https://saifmohammad.com/WebPages/AffectIntensity.htm)
This dataset provides a list of 15,256 English words with emotion intensity scores for 8 emotion categories: `anger`, `anticipation`, `disgust`, `fear`, `joy`, `sadness`, `surprise`, and `trust`. The intensity scores range from 0 to 1, with higher values indicating stronger emotion.

## Methods
### Preprocessing
We apply a series of transformation to the movie summaries, including tokenization, lemmatization, removing punctuations, and converting to lowercase. These preprocessing steps ensures proper recognition of words associated with emotions that are defined in the emotion lexicon.

### Genre-Emotion Analysis
This analysis is conducted on a per genre basis, focusing on the top-10 prevalent genres while assuming the first genre associated with each genre as the main genre.

#### Emotion Quantification 
Two metrics are employed for evaluating emotions in movie plot summaries and titles:

* Emotion intensity vector: This metric is derived from the [NRC Emotion Intensity Lexicon](https://saifmohammad.com/WebPages/AffectIntensity.htm). We compute the emotion intensity vector for each summary by aggregating the intensities of each lexicon word and normalization, such that the entries in the vector indicate relative strength of the 8 emotions.
* Sentiment score: We use the [TextBlob](https://textblob.readthedocs.io/en/dev/) library for calculating sentiment scores for movie summaries and titles. The sentiment scores are further discretized into 5 ordinal categories, which are used as labels in machine learning.

#### Cluster Analysis
To identify the prevalent emotions and typical emotion intensity patterns for each genre, we perform cluster analysis on emotion intensity vectors. This analysis answers our first research question.

#### Genre Definition by Emotion
The idea of linking genres with emotions is further extended by creating a definition of each genre with emotional tags. For this analysis we utilize the [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) pipeline, which contains 27 emotion categories and provides a pretrained BERT model for inferencing.

#### Emotion along the Plot
We are interested to explore the ups and downs in movie sentiments along the storyline, across different genres. The changes in sentiment across different parts of the movie are tracked by aggregating sentiment scores over consecutive groups of sentences or by calculating the moving average of sentiment scores.

#### Time-Based Emotional Trends
As people's tastes evolve, the preferred emotion might shift over the years. For example, in an era dominated by socio-political unrest, audiences might lean towards uplifting films as an escapade. We exploit this idea by two approaches:

* Time-series analysis: Develop line plots for genre-specific emotion intensities over time.
* Cohort analysis: Segment the data into cohorts based on movie release years or periods and compare emotion intensity trends across cohorts.

### Emotion as a Predictor
#### Machine Learning
We adopt decision tree and random forest models for predicting missing box office revenue data. We start with genre and sentiment score as the only input features, and iteratively add to the complexity of the model by considering other variables, e.g., movie runtime, main character gender, language, and country. We apply discretization to continuous input features, and assess the prediction performance using both label encoding and one-hot encoding, and select the method that yields better performance. The aim of this analysis is to find a subset of variables that enables us to determine the success of a movie.

#### Regression Analysis
We conduct multivariate regression per genre on box office revenue with emotion intensities as independent variables. With normalized intensity vectors, the regression coefficients can indicate the relative importance of emotions to movie revenue success for each genre. 

## Proposed Timeline
* 18.11.2023 - 24.11.2023: Cluster analysis, emotion along the plot

* 25.11.2023 - 01.12.2023: Homework H2

* 02.12.2023 - 08.12.2023: Genre definition by emotion

* 09.12.2023 - 15.12.2023: Finalize visualization, develop data story

* 16.12.2023 - 22.12.2023: Finalize data story and notebook

## Organization within the Team
Berta: Emotion along the plot

Inès: Sentiment score calculation, machine learning

Mya: Genre-gender analysis, sentiment score analysis

Fernando: Emotion intensity vector calculation, time-series analysis

Xiaocheng: Regression analysis, cluster analysis