# ✨✨ WE NEED A TITLE ✨✨ 

*Badafixm01: Berta Céspedes, Inès Kahlaoui, Mya Lahjouji, Fernando Meireles, Xiaocheng Zhang*

## Abstract

How do movie genres differ in the realm of emotions? We take a historical look over a century-long period, and explore the intentional emotional injections by film producers, be they thrill, romance, fear, or joy. Leveraging the movie plot summaries and their corresponding metadata, we use NLP techniques to extract emotions and sentiment scores from the summaries. Our exploration focuses on the shifts in emotional patterns in the cinematic world as genres evolve and fade in the ever-progressing cinematic landscape. We hypothesize that certain emotions, like joy or fear, could persist across generations within specific genres, while others may be sensitive to the global societal background. By investigating the emotion-genre-generation trio, this project aims to reveal the underlying emotional *décor* that shapes the genres and the role of social generations on this dynamic. This investigation could paint a picture of society's evolving emotional needs and preferences throughout different eras.


## ❓ Research Question
In our project, we are looking to solve the following questions
*   What are the typical patterns of emotion intensities for the most prevalent genres? Which emotions are more common in which movie genres?
*   How do movie sentiments compare over different time periods?
*   Taking specific worldwide events, can we identify particular emotions? In other words, did worldwide events impact the movie world?


## 📒 Datasets
### [CMU Movie Summary Corpus](https://www.cs.cmu.edu/~ark/personas/)
This is the provided dataset for the analysis. Our research questions mostly concern the movie summary, genre, release year, and country data. During the initial filtering of the dataset, we found 42,207 movies with available summary and metadata. Since the dataset contains a total of 364 genres and some of them may not be meaningful, we only consider the first genre associated with each movie as the main genre.


### [NRC Emotion Intensity Lexicon](https://saifmohammad.com/WebPages/AffectIntensity.htm)
This dataset provides a list of 15,256 English words with human-annotated emotion intensity scores for 8 emotion categories: `anger`, `anticipation`, `disgust`, `fear`, `joy`, `sadness`, `surprise`, and `trust`. The intensity scores range from 0 to 1, with higher values indicating stronger emotions.


## 📊 Methods
### Part I: Getting to Know the Dataset

#### Preprocessing
We apply a series of transformation to the movie summaries, including tokenization, lemmatization, removing punctuations, and converting to lowercase. These preprocessing steps ensures proper recognition of words associated with emotions that are defined in the emotion lexicon.

#### Exploratory Data Analysis
We conducted exploratory data analysis on various aspects of the movie metadata, including number of movie releases, genres, countries, languages, runtime, and release date and period. To uncover interesting trends and patterns, we employed visualization techniques such as bar plots, stacked bar plots, and line plots. These visualizations help identify interesting trends and patterns for further analysis.

---

### Part II: Emotion Quantification

#### Emotion Intensity Vector
This metric is derived from the [NRC Emotion Intensity Lexicon](https://saifmohammad.com/WebPages/AffectIntensity.htm). Given an emotion, the intensity scores range from 0 to 1, indicating the lowest to the highest amount of the emotion. The emotion intensity vector for each summary is computed by aggregating the intensities of each lexicon word and normalization, such that the entries in the vector indicate the relative strength of the 8 emotions.

#### Sentiment Score
We use the [TextBlob](https://textblob.readthedocs.io/en/dev/) library for calculating polarity sentiment scores for movie summaries. The sentiment scores lie between [-1, 1], where -1 defines a negative sentiment and 1 defines a positive sentiment.

---

### Part III: Genre-Emotion Analysis

#### Cluster Analysis
Can we easily delineate genres according to the emotional patterns? To find the answer, we perform cluster analysis on emotion intensity vectors using dimensionality reduction techniques, t-SNE and PCA, followed by the K-means algorithm. The clustering enables the grouping of emotion intensity vectors based on their proximity, efficiently identifying boundaries in the data. This approach allows us to validate whether emotional intensities can serve as a signature for genres.

#### Regression Analysis
To what degree can emotional intensities and sentiment help shape the genre? We seek the answer in regression analysis. We build Ordinary Least Squares models with emotion intensities and sentiment score as independent variables. Paired matching is performed on country, language, generation, and runtime to eliminate confounding factors. The significance of the regression coefficients are examined by *t*-tests.

---

### Part IV: Time-based Analysis
As people's tastes evolve, the prevalent emotions in the cinematic world might shift over the years. For example, during an era dominated by socio-political unrest, movies with tension and angst might emerge to reflect societal issues. We exploit this idea by splitting the data into generations based on movie release years. The generations considered in the analysis include: The Lost Generation (born between 1883 and 1900), The Greatest Generation (1901–1927), The Silent Generation (1928–1945), The Baby Boomers (1946–1964), Generation X (1965–1980), The Millennials (1981–1996), Generation Z (1997–2009), Generation Alpha (2010–2024).

#### Emotion Intensity Trends
We construct time series of emotional intensity vectors with two types of visualizations:
*   Stacked bar plot for emotional intensities over generations.
*   Line plot for genre-specific emotional intesities over years.

#### Emotion Word Cloud Development
Having seen the relationship between generations and basic emotions, such as anger, joy and sadness, we delve deeper into this relationship by using the [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) pipeline, which contains 28 emotion categories and provides a pretrained BERT model for inferencing. We build word clouds of the 28 emotions for a control period 1997-2001 and three historical events: World War II, Cold War, and 9/11. This analysis helps answer whether significant historical events are reflected in the emotions of the movies produced in the corresponding time periods.


## ⏱️ Proposed Timeline
| Start Date  | End Date    | Task                                         |
|-------------|-------------|----------------------------------------------|
| 18.11.2023  | 01.12.2023  | Homework H2                                  |
| 02.12.2023  | 08.12.2023  | Cluster analysis                             |
| 09.12.2023  | 12.12.2023  | Emotion word cloud                           |
| 13.12.2023  | 15.12.2023  | Finalize visualization, develop data story   |
| 16.12.2023  | 22.12.2023  | Finalize data story and notebook             |


## 👫 Organization within the Team
| Name       | Contributions                                               |
|------------|-------------------------------------------------------------|
| Berta      |(1) Run models to construct word clouds                      |
|            |(2) Perform country-based exploratory analysis               |
| Inès       |(1) Plan out data analysis and visualization                 |
|            |(2) Develop data story                                       |
| Mya        |(1) Perform data preprocessing                               |
|            |(2) Perform genre and generation-based exploratory analysis  |
|            |(3) Visualize sentiment score related findings               |
| Fernando   |(1) Calculate emotion intensity vector                       |
|            |(2) Perform cluster analysis                                 |
|            |(3) Visualize emotion intensity related findings             |
| Xiaocheng  |(1) Perform regression analysis                              |
|            |(2) Perform matching for generation-based analysis           |
|            |(3) Document README and final notebook                       |
