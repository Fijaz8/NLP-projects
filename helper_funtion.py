import re
import nltk
from nltk.stem import PorterStemmer
ps=PorterStemmer()
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
stopwords=nltk.corpus.stopwords.words('english')
import numpy as np
def clean_text(text):
  stopwords=nltk.corpus.stopwords.words('english')
  ps = PorterStemmer()
  # Define a pattern to remove hyperlinks
  hyperlink_pattern = r'\b\w*http[s]?://\S*\b|\b\w*www\.\S*\b'

  # Remove hyperlinks
  text = re.sub(hyperlink_pattern, '', text)

  # Remove any email addresses if needed (original pattern)
  email_pattern = r'\w+\.\w+@\w+\.\w+'
  text = re.sub(email_pattern, '', text)

# Remove mentions and hashtags
  mention_hashtag_pattern = r'\b[@#]\w+\b'
  text = re.sub(mention_hashtag_pattern, '', text)
    

  text=re.sub(r'[^A-Z a-z]','',text)
  text = re.sub(r'[@#_]', ' ', text)
  text=text.lower()
  text=text.split()
  text=[ps.stem(word) for word in text if word not in stopwords]
  text=" ".join(text)
  return text

from nltk.tokenize import word_tokenize
def process_tweet(tweets):
  tokenized_tweets = []  # Initialize an empty list to store tokenized tweets
  for tweet in tweets:
      tokenized_words = word_tokenize(tweet)
      tokenized_tweets.append(tokenized_words)
  return tokenized_tweets

def build_freqs(tweeTtok, ys):
    """Build frequencies.
    Input:
        tweeTtok: a list of tweets
        ys: an m x 1 array with the sentiment label of each tweet
            (either 0 or 1)
    Output:
        freqs: a dictionary mapping each (word, sentiment) pair to its
        frequency
    """
    # Convert np array to list since zip needs an iterable.
    # The squeeze is necessary or the list ends up with one element.
    # Also note that this is just a NOP if ys is already a list.
    yslist = np.squeeze(ys).tolist()

    # Start with an empty dictionary and populate it by looping over all tweets
    freqs = {}
    for y, tweet in zip(yslist, tweeTtok):
        for word in process_tweet(tweet):
            if isinstance(word, list):
                for w in word:
                    pair = (w, y)
                    if pair in freqs:
                        freqs[pair] += 1
                    else:
                        freqs[pair] = 1
            else:
                pair = (word, y)
                if pair in freqs:
                    freqs[pair] += 1
                else:
                    freqs[pair] = 1
    return freqs

