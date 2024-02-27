import time
start_time = time.time()


import os
import re
import html
import nltk
import gensim
import pickle
import pandas as pd
import pyLDAvis.gensim as gensimvis
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
import random
from nltk.tokenize import word_tokenize
from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora
import pyLDAvis
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import plotly
import hdbscan


df = pd.read_csv("filtered_data/alltweetsrelatedtonetwork.csv")
df['city_identifier'] = df['city_identifier'].astype(str).str.replace('.csv', '')

# merge to get city names for df
cwd = os.getcwd()
file_city = os.path.join(cwd,"merged_sentiment","merge_city.csv")
df1 = pd.read_csv(file_city, encoding='utf-8') 
df1['city_identifier'] = df1['city_identifier'].astype(str)
df = pd.merge(df, df1, how='left', left_on=['city_identifier'], right_on = ['city_identifier'])


cities = [
    "New York", "Miami", "Chicago", "Houston", "Washington", "Austin",
    "Atlanta", "Boston", "Dallas", "Los Angeles", "Phoenix", "Seattle",
    "Denver", "San Diego", "San Francisco", "Portland", "Las Vegas",
    "Baltimore", "Charlotte", "Philadelphia", "Nashville", "Detroit",
    "Sacramento", "Jacksonville", "San Antonio", "Mesa", "San Jose",
    "Raleigh", "Columbus", "Louisville", "Indianapolis", "Milwaukee",
    "Fort Worth", "Arlington", "Tucson", "Oklahoma City", "Albuquerque",
    "Memphis", "Omaha", "Long Beach", "Kansas City", "Colorado Springs",
    "Fresno", "Tulsa", "Wichita", "Virginia Beach", "El Paso",
    "Bakersfield", "Oakland", "Minneapolis"
]

# Function to remove URLs, city names, 'los', 'angeles', and 'real estate'
def preprocess_text(text):
    # Decode HTML entities
    text = html.unescape(text)
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove the phrase "real estate" (added line)
    text = re.sub(r'\breal\s*estate\b', '', text, flags=re.IGNORECASE)
    # Remove city names and other specified terms
    city_pattern = r'\b' + r'\b|\b'.join(re.escape(city) for city in cities) + r'\b'
    text = re.sub(city_pattern, '', text, flags=re.IGNORECASE)
    # Tokenize the text to filter out stopwords
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words and word.isalpha()]
    # Rejoin words into a single string
    text = ' '.join(filtered_words)
    # Optional: Remove '<' and '>' if they still exist (might not be necessary after previous steps)
    text = text.replace('<', ' ').replace('>', ' ')
    return text



# df_subset = df.sample(n=100000)
# Assuming `tweets` is a list of preprocessed tweets:
tweets = [preprocess_text(tweet) for tweet in df['text']]

##############################################
# Randomly sample tweets from the list
sample_size = 300000 # Adjust the sample size as needed
sample_tweets = random.sample(tweets, sample_size)
len(sample_tweets)

###################################
# Configure the HDBSCAN model
hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=300, 
                                min_samples=30,
                                metric='euclidean', 
                                cluster_selection_method='eom')

# Initialize BERTopic with the custom HDBSCAN model
topic_model = BERTopic(hdbscan_model=hdbscan_model)


# Fit the model to your tweets
topics, probabilities = topic_model.fit_transform(sample_tweets)

# Optional: Adjust parameters as needed, e.g., nr_topics to reduce the number of topics
# topic_model.update_topics(tweets, topics, n_gram_range=(1, 2))

topic_info = topic_model.get_topic_info()

# For example, to print the content of the first cell
topic_info.to_csv("topic_info_300000_c300s30.csv")

end_time = time.time()
total_time = end_time - start_time
print(f"Total execution time: {total_time} seconds")
