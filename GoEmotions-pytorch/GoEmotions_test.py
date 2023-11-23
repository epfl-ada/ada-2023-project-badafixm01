import numpy as np
from transformers import BertTokenizer, AutoTokenizer
from model import BertForMultiLabelClassification
from multilabel_pipeline import MultiLabelPipeline
from pprint import pprint
import torch
import pandas as pd
import ast
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import re
import nltk
from collections import Counter
from transformers import pipeline

device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Open and read the text file
# with open(R"C:\Users\Berta\Desktop\EPFL\ADA\PROJECT\MovieSummaries\plot_summaries.txt", 'r', encoding='utf-8') as file:
with open(R"C:\Users\Berta\Desktop\EPFL\ADA\PROJECT\MovieSummaries\plot_summaries.txt", 'r', encoding='utf-8') as file:    
    content_summaries = file.read()

# Split the content into individual summaries based on "movie_id"
summaries = content_summaries.split('\n')

# Remove empty lines if any
plot_summaries = [summary.strip() for summary in summaries if summary.strip()]

movie_ids = []
summary_texts = []

# Split the plot summaries into movie IDs and summary text
for summary in plot_summaries:
    parts = summary.split('\t', 1)  # Split at the first space character
    if len(parts) == 2:
        movie_id, summary_text = parts
        movie_ids.append(int(movie_id))
        summary_texts.append(summary_text)

summaries_df = pd.DataFrame({'Movie ID': movie_ids, 'Plot': summary_texts})

column_names = [
    'Movie ID',
    'Freebase ID',
    'Movie Title',
    'Release Date',
    'Box Office',
    'Runtime',
    'Language',
    'Country',
    'Genre'
]

df = pd.read_csv(R"C:\Users\Berta\Desktop\EPFL\ADA\PROJECT\MovieSummaries\movie.metadata.tsv", delimiter='\t', header=None, names=column_names)
df = df[~(df['Genre']=='{}').values]

def extract_first_genre(genre_str):
    genre_dict = ast.literal_eval(genre_str)
    return next(iter(genre_dict.values()))

df['Genre'] = df['Genre'].apply(extract_first_genre)
df['Language'] = df['Language'].apply(lambda x: ', '.join([value.split()[0] for key, value in ast.literal_eval(x).items()]))
df['Country'] = df['Country'].apply(lambda x: ', '.join([value for key, value in ast.literal_eval(x).items()]))

metadata = df[df['Movie ID'].isin(movie_ids)]
summaries_df = summaries_df[summaries_df['Movie ID'].isin(metadata['Movie ID'])]

movie_data = pd.merge(metadata, summaries_df, on='Movie ID')

# Preparing data for tokenisation:

# Note: same summary and title for this entries, but different release year! They also cause problems with manual tokenisation, so they will be removed for the moment. Further analysis needed.
# movie_data.iloc[9724]
# movie_data.iloc[25327]
# movie_data.iloc[29447]
# movie_data.iloc[33197]

# Other entries which caused problems with tokenizer:
# movie_data.iloc[16550]
# movie_data.iloc[29414]
# movie_data.iloc[30229]
# movie_data.iloc[34493]

plots_need_analsys = movie_data.iloc[[9724, 16550, 25327, 29414, 29447, 30229, 33197, 34493]]
movie_data = movie_data.drop([9724, 16550, 25327, 29414, 29447, 30229, 33197, 34493])

# Count 10 most recurrent Genres:
genre_occurrences = movie_data['Genre'].value_counts()
top_10_genres = genre_occurrences[:10].index.to_numpy()
movie_data_top10 = movie_data[movie_data['Genre'].isin(top_10_genres)]


# Use pre-trained NLP model which takes input text and gives back k emotions

# Load the pretrained GoEmotions model and tokenizer
model_name = "monologg/bert-base-cased-goemotions-original"
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

import nltk
# from transformers import AutoTokenizer

# Manual tokenization with max_len=512 to avoid sentences splitting in half and allow mapping emotions to each plot summary:

nltk.download('punkt')

def sentence_aware_split(text, max_length, tokenizer):
    # Split text into sentences
    sentences = nltk.tokenize.sent_tokenize(text)

    chunks = []
    current_chunk = ""
    for sentence in sentences:
        # Check if adding this sentence would exceed the max_length
        potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
        potential_chunk_tokenized = tokenizer.tokenize(potential_chunk)
        
        if len(potential_chunk_tokenized) <= max_length - 2:  # -2 for [CLS] and [SEP]
            current_chunk = potential_chunk
        else:
            # Add the current_chunk to chunks and start a new one
            chunks = chunks + [current_chunk]
            current_chunk = sentence

    # Don't forget to add the last chunk
    if current_chunk:
        chunks = chunks + [current_chunk]

    return chunks

max_length = 512  # Adjust based on your model's max length

texts = movie_data_top10['Plot'].values.tolist()
tokenized = []
chunk_to_text_mapping = {}

for i, text in enumerate(texts):
    
    chunked_text = sentence_aware_split(text, max_length, tokenizer)

    for chunk in chunked_text:
        chunk_to_text_mapping[chunk] = i

    tokenized = tokenized + chunked_text

# Save summary chunks and chunks to plot mapping for future runs:

def save_list_to_file(list_of_strings, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for i, string in enumerate(list_of_strings):
            file.write(string + '\n')

def save_dict_to_file(dictionary, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for key, value in dictionary.items():
            file.write(f'{key}: {value}\n')

save_list_to_file(tokenized, R'C:\Users\Berta\Desktop\EPFL\ADA\PROJECT\ada-2023-project-badafixm01\GoEmotions-pytorch\tokenized_plots.txt')
save_dict_to_file(chunk_to_text_mapping, R'C:\Users\Berta\Desktop\EPFL\ADA\PROJECT\ada-2023-project-badafixm01\GoEmotions-pytorch\chunk_mapping.txt')

# Cell to load files to avoid re-running manual tokenisation cell:
with open(R"C:\Users\Berta\Desktop\EPFL\ADA\PROJECT\MovieSummaries\tokenized_plots.txt", 'r', encoding='utf-8') as file:    
    tokenized_plots = file.read()

# Split the content into individual summaries based on "movie_id"
tokenized = tokenized_plots.split('\n')

with open(R"C:\Users\Berta\Desktop\EPFL\ADA\PROJECT\MovieSummaries\chunk_mapping.txt", 'r', encoding='utf-8') as file:    
    chunk_mapping = file.read()

# Split the content into individual summaries based on "movie_id"
text_maps = chunk_mapping.split('\n')

tokenizer = BertTokenizer.from_pretrained("monologg/bert-base-cased-goemotions-original")
model = AutoModelForSequenceClassification.from_pretrained("monologg/bert-base-cased-goemotions-original")

goemotions = pipeline(
        model=model, 
        tokenizer=tokenizer, 
        task="text-classification",
        return_all_scores=True,
        function_to_apply='sigmoid',
    )

# save_list_to_file(tokenized, R'C:\Users\Berta\Desktop\EPFL\ADA\PROJECT\MovieSummaries\tokenized_plots.txt')
# save_dict_to_file(chunk_to_text_mapping, R'C:\Users\Berta\Desktop\EPFL\ADA\PROJECT\MovieSummaries\chunk_mapping.txt')

# with open(R"C:\Users\Berta\Desktop\EPFL\ADA\PROJECT\MovieSummaries\tokenized_plots.txt", 'r', encoding='utf-8') as file:    
#     tokenized_plots = file.read()

# # Split the content into individual summaries based on "movie_id"
# summaries = tokenized_plots.split('\n')

# with open(R"C:\Users\Berta\Desktop\EPFL\ADA\PROJECT\MovieSummaries\chunk_mapping.txt", 'r', encoding='utf-8') as file:    
#     chunk_mapping = file.read()

# # Split the content into individual summaries based on "movie_id"
# text_maps = chunk_mapping.split('\n')

# print('hey')


# inputs = tokenizer(tokenized, padding = True, return_tensors = 'pt')

# with torch.no_grad():
#     outputs = model(**inputs)

# # Get predicted probabilities for each emotion label
# predicted_probabilities = outputs.logits.softmax(dim=1)

# # You can access the emotion labels from the model's config
# emotion_labels = list(model.config.id2label.values())


# # Convert tensor to numpy array
# predicted_probabilities = predicted_probabilities.numpy()

# # Get the top-k predicted emotions (e.g., top 3)
# k = 3
# top_k_emotions = [[emotion_labels[i] for i in np.argsort(movie[0])[::-1][:k]] for movie in predicted_probabilities]

# # for movie in range(len(movie_data)):




# # movie_dict = {}
# # for i, summary in enumerate(summary_texts):
# #     # sentiment_count = Counter(sent_list)
# #     # majority_sent = sentiment_count.most_common(1)[0][0]
# #     movie_dict[movie_ids[i]] = {'summary': summary}




# # tokenizer = BertTokenizer.from_pretrained("monologg/bert-base-cased-goemotions-original")
# # model = BertForMultiLabelClassification.from_pretrained("monologg/bert-base-cased-goemotions-original").to(device)

# # goemotions = MultiLabelPipeline(
# #     model=model,
# #     tokenizer=tokenizer,
# #     threshold=0.3
# # )

# # text = summary_texts[0]

# # pprint(goemotions(text))





# print("Top Emotions:", top_k_emotions)

# # movie_dict = {}
# # for i, (summary, sent_list) in enumerate(zip(summary_texts, sentiment_classifier)):
# #     sentiment_count = Counter(sent_list)
# #     majority_sent = sentiment_count.most_common(1)[0][0]
# #     # majority_sentiment = sentiment_counts.most_common(1)[0][0]
# #     movie_dict[movie_ids[i]] = {'summary': summary, 'sentiments': sentiment_classifier, 'overall_sentiment': majority_sent}

# print('done!')