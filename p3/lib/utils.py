import re
import string
import importlib.util
import pycountry
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from textblob import TextBlob
from tqdm.auto import tqdm
from collections import Counter
from wordcloud import WordCloud


def contains_nan(lst):
    return any(pd.isna(item) for item in lst)

def is_single_nan(lst):
    return len(lst) == 1 and pd.isna(lst[0])

def extract_language(language_data):
    language_names = []
    pattern = r'"([^"]+)"\s*:\s*"([^"]+)"'
    matches = re.findall(pattern, language_data)
    for match in matches:
        language_names.append(match[1].split(' ')[0])  # Extract the language name
    return ','.join(language_names)

def extract_countries(country_data):
    country_names = []
    pattern = r'"([^"]+)"\s*:\s*"([^"]+)"'
    matches = re.findall(pattern, country_data)
    for match in matches:
        country_names.append(match[1])  # Extract the country name
    return ','.join(country_names)

def extract_genres(genre_data):
    genre_names = []
    pattern = r'"([^"]+)"\s*:\s*"([^"]+)"'
    matches = re.findall(pattern, genre_data)
    for match in matches:
        genre_names.append(match[1])  # Extract the genre name
    return ','.join(genre_names)

def extract_first_genre(genre_list):
    if genre_list:
        return genre_list[0]
    else:
        return None 
 
def extract_release_year(date_str):
    """
    Extract the release year from the date.
    """
    try:
        # Attempt to extract the year from the 'YYYY-MM-DD' format
        return pd.to_datetime(date_str).year
    except (ValueError, TypeError):
        try:
            # Attempt to extract the year from 'YYYY' format
            return int(date_str)
        except ValueError:
            return None  # Return None for invalid or missing dates
        
def get_generation(year, generations):
    """
    Determine the generation of a given year.
    """
    for _, row in generations.iterrows():
        if row['Start Year'] <= year <= row['End Year']:
            return row['Generation']
    return "Unknown Generation"
        
def preprocess_summary(text):
    """
    Tokenize, lemmatize, remove stopwords and punctuations from an input text.
    
    Parameters
    ----------
    text: str, input text
    
    Returns
    -------
    str, preprocessed text
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words('english')
    
    tokens = nltk.word_tokenize(text)
    text = [word for word in tokens if word not in string.punctuation]
    
    return " ".join([lemmatizer.lemmatize(word.lower()) for word in text if word.lower() not in stop_words])

def preprocess_summaries(readpath, savepath=None):
    """
    Preprocess all movie summaries.
    
    Parameters
    ----------
    readpath: str, path to original summaries (.txt)
    savepath: str, path to preprocessed summaries (.csv)
    
    Returns
    -------
    original_summaries: list, original summaries
    df: pandas.DataFrame, preprocessed summaries with movie_id as index
    """
    with open(readpath, encoding='utf-8') as f:
        content = f.readlines()
    original_summaries = [x.strip() for x in content] 
    summaries = [preprocess_summary(d).split() for d in original_summaries]
    summaries = {summary[0]: " ".join(summary[1:]) for summary in summaries}
    
    df = pd.DataFrame.from_dict(summaries, orient='index', columns=['text'])
    df.index = df.index.astype('int64').rename('movie_id')
    
    if savepath is not None:
        df.to_csv(savepath)
    return original_summaries, df

def clean_text(text):
    # Remove all '{{XX}}' patterns, HTML tags if any & extra white spaces
    text = re.sub(r'\{\{[^}]*\}\}', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove each specific typos
    specific_strings = [r'\{\{Plot\|date', r'\{\{plot\|date', r'\{\{plot\|section\|date', r'\{\{Plot\|section\|date']
    for string in specific_strings:
        text = re.sub(string, '', text)

    return text

def plot_sentiment_score(text_list):
    """
    Get compound sentiment score for text.
    """
    text = " ".join(text_list)  # Convert list of words to a single string
    return TextBlob(text).sentiment.polarity

def standardize(df, features):
    for feature in features:
        df[feature] = (df[feature] - df[feature].mean()) / df[feature].std()
    return df

def get_control_treated_split(df, column):
    control = df.groupby(column).get_group(0).copy()
    treated = df.groupby(column).get_group(1).copy()
    return control, treated

def plot_control_treated_feature(df, column, value, dtype):
    data = df[["Wikipedia Movie ID", column, value]]\
           .pivot(index="Wikipedia Movie ID", columns=column, values=value)
    plt.figure(figsize=(4, 3))
    sns.boxplot(data=data)
    plt.xlabel("")
    plt.ylabel(value)
    ticks, _ = plt.xticks()
    plt.xticks(ticks=ticks, labels=["control", "treated"])
    plt.title(f"{value} for {column.split('_')[1]} {dtype} matching")
    plt.show()

def country_to_iso(country):
    """
    Convert country names to ISO code.
    """
    special_cases = {
        "United States of America": "USA",
        'German Democratic Republic': 'DEU', 
        'South Korea': 'KOR',
        'West Germany': 'DEU',
        'England': 'GBR',
        'Czech Republic': 'CZE',
        'Nazi Germany': 'DEU',
        'Weimar Republic': 'DEU',
        'Iran': 'IRN',
        'Taiwan': 'TWN',
        'Russia': 'RUS',
        'Scotland': 'GBR',
        'Venezuela': 'VEN',
        'Vietnam': 'VNM',
        'Republic of Macedonia': 'MKD',  
        'Burma': 'MMR', 
        'Kingdom of Great Britain': 'GBR',
        'Democratic Republic of the Congo': 'COD',
        'Uzbek SSR': 'UZB',
        'Northern Ireland': 'GBR',
        'Wales': 'GBR',
        'Bolivia': 'BOL',
        'Ukrainian SSR': 'UKR',
        'Palestinian territories': 'PSE',
        'Georgian SSR': 'GEO'
}
    if country in special_cases:
        return special_cases[country]
    # otherwise:
    try:
        return pycountry.countries.get(name=country).alpha_3
    except:
        return None  # Returns None if the country was not found

def iso_to_country(iso_code):
    """
    Convert ISO codes to country names.
    """
    try:
        return pycountry.countries.get(alpha_3=iso_code).name
    except:
        return None  # Returns None if the ISO code was not found
    
def matching_countries_across_generations(df, match_generations):
    """
    Return a DataFrame with same number of movies across generations from each country.
    """
    df = df[df["Generation"].isin(match_generations)]
    group_cols = ["Generation", "Movie Countries"]
    
    # find common countries across all generations
    common_countries = set(df["Movie Countries"])
    for generation in match_generations:
        common_countries = common_countries.intersection(set(df[df["Generation"] == generation]["Movie Countries"]))
    
    # downsample to include the same number of movies for each country
    matched_dfs = []
    target_size = df.groupby(["Generation", "Movie Countries"]).size().reset_index(name="count")
    
    min_count = {}
    for country in common_countries:
        count = target_size[target_size["Movie Countries"] == country].groupby("Generation")["count"].min()
        min_count[country] = count.min()
    
    for country in common_countries:
        country_df = df[df["Movie Countries"] == country]
        matched_df = country_df.groupby(group_cols).apply(lambda x: x.sample(min(len(x), min_count[country]), random_state=42))
        matched_dfs.append(matched_df)
    
    result = pd.concat(matched_dfs)
    return result.reset_index(drop=True)

def load_module(module_name, file_path):
    """
    Load a module from a given filepath.
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def split_text_in_groups(text, group_size=3):
    """
    Split plots in group_size sentences.
    """
    # Split text into sentences
    sentences = text.split('. ')
    groups = []
    
    for i in range(0,len(sentences), group_size):
        if i + group_size > len(sentences):
            current_group_size = len(sentences) - i
        else:
            current_group_size = group_size

        # Select sentences for the current group
        current_group = sentences[i:i+current_group_size]
        # Join the sentences in the current group
        group = '. '.join(current_group)
        # Add a period at the end if it's not already there
        if len(current_group) > 0 and not group.endswith('.'):
            group += '.'
        # Add the group to the list of groups
        groups.append(group)
    return groups

def create_batches(data, batch_size):
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

def train(batches, model):
    emotions = []
    for i, batch in enumerate(tqdm(batches, desc="Processing Batches")):
        output = model(batch)
        
        # Save data
        top_emotions = [', '.join(text['labels']) for text in output]
        emotions_df = pd.DataFrame({'Top Emotions': top_emotions})
        emotions_df.to_csv('top_emotions.csv', mode='a', index=False, header=False)
        
        emotions.extend(top_emotions)
    return emotions

global_index = 0

def get_strings(n, emotions):
    global global_index
    # Select the emotions starting from the current global index
    results = emotions[global_index:global_index+n]
    # Update the global index for the next call
    global_index += n
    return results

def split_and_flatten_emotions(emotions_list):
    flattened_list = []
    for emotion_combo in emotions_list:
        # Splitting by comma and stripping whitespace
        split_emotions = [emotion.strip() for emotion in emotion_combo.split(',')]
        flattened_list.extend(split_emotions)
    return flattened_list

def create_word_clouds(series, year_ranges):
    # Iterate over the Series
    for index, (label, emotions_list) in enumerate(series.items()):
        # Convert the list of emotions to a single string
        frequencies = Counter(emotions_list)

        # Create a WordCloud object
        wordcloud = WordCloud(width=1000, height=800, 
                            background_color='white', 
                            min_font_size=10).generate_from_frequencies(frequencies)

        # Plot the WordCloud image                        
        plt.figure(figsize=(10, 8), facecolor=None) 
        plt.imshow(wordcloud) 
        plt.axis("off") 
        plt.tight_layout(pad=0)
        plt.title(f"Emotion word cloud for {label} ({year_ranges[index][0]}-{year_ranges[index][1]})", fontsize=16)

        plt.show()

def in_range(year, ranges):
    """
    Check if a year falls within any of the ranges.
    """
    return any(start <= year <= end for start, end in ranges)