import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import contractions  # https://github.com/kootenpv/contractions
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import regex
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_validate
import warnings
import tensorflow as tf
from keras import layers
import lightgbm as lgb
from keras.layers import Dense, Dropout, Bidirectional
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Bidirectional
from keras.layers import Embedding
from keras.layers import GlobalAvgPool1D
import keras_tuner

warnings.filterwarnings(action='ignore', category=UserWarning)

# ----------------------------------------------------------------------------------------------------------------------

data = pd.read_csv('C:/Users/ata-d/OneDrive/Masaüstü/ML/Datasets/spotify_reviews.csv')


# ----------------------------------------------------------------------------------------------------------------------

# Check datatypes. Time_submitted does not look okay
print(data.dtypes)

# Correcting the wrong datatype (Turning object time column into datetime format)
data.Time_submitted = pd.to_datetime(data.Time_submitted)

# Check for missing data. Reply column has mostly nan values, but it is okay for now
print(data.isna().sum())

# Create a new column based on length of the reviews
data['length_of_text'] = [len(i.split(' ')) for i in data['Review']]

# Let's look at the summary of the dataset
print(data.head())

# Let's look at the summary of the dataset
print(data.describe())

# ----------------------------------------------------------------------------------------------------------------------

# Distribution of the Length of the Reviews (Length of Text greater than 120 is neglected)
'''''''''
fig = px.histogram(data[data['length_of_text'] <= 120].length_of_text, marginal='box',
                   labels={"value": "Length of the Reviews"})
fig.update_traces(marker=dict(line=dict(color='#000000', width=0.25)))
fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})
fig.update_layout(title_text='Distribution of the Length of the Reviews',
                  title_x=0.5, title_font=dict(size=18), showlegend=False)
fig.show()
'''''''''

# Text length mean, std, number of ratings and total words for each Ratings
print(data.groupby('Rating').length_of_text.agg({'mean', 'std', 'count', 'sum'}))

# Distribution of the Length of the Reviews by their Ratings (Length of Text greater than 120 is neglected)
'''''''''
fig = px.histogram(data[data['length_of_text'] <= 120].length_of_text, marginal='box',
                   labels={"value": "Length of the Reviews"},
                   color=data[data['length_of_text'] <= 120].Rating)
fig.update_traces(marker=dict(line=dict(color='#000000', width=0.25)))
fig.update_layout(title_text='Distribution of the Length of the Reviews by their Ratings',
                  title_x=0.5, title_font=dict(size=18))
fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})
fig.show()
'''''''''

# Frequency of the Ratings
'''''''''
fig = px.histogram(data, x='Rating', color='Rating')
fig.update_traces(marker=dict(line=dict(color='#000000', width=0.5)))
fig.update_layout(title_text='Frequency of the Ratings',
                  title_x=0.5, title_font=dict(size=18))
fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})
fig.update_layout(bargap=0.2)
fig.show()
'''''''''

# Number of Reviews Over Time
'''''''''
dailyNumOfReviews = data.resample('d', on='Time_submitted').size().sort_index()

fig = go.Figure()
fig.add_trace(go.Scatter(y=dailyNumOfReviews, x=dailyNumOfReviews.index,
                         mode='markers+text+lines', name=f"Number of Reviews", line=dict(width=3)))

fig.update_layout(
    yaxis=dict(title_text="Number of Reviews", titlefont=dict(size=15)),
    xaxis=dict(title_text="Date", titlefont=dict(size=15)),
    title={'text': f"Number of Reviews Over Time",
           'x': 0.5})
fig.update_traces(marker=dict(line=dict(color='#000000', width=0.75)))
fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})
fig.show()
'''''''''

# Number of Reviews Over Time for each Rating
'''''''''
dailyNumOfReviewsEachRatings = data.groupby('Rating').resample('d', on='Time_submitted').size().sort_index().T

fig = go.Figure()
for k in range(len(dailyNumOfReviewsEachRatings.columns)):
    fig.add_trace(go.Scatter(y=dailyNumOfReviewsEachRatings[dailyNumOfReviewsEachRatings.columns[k]],
                             x=dailyNumOfReviewsEachRatings.index,
                             mode='lines',
                             name=f"Number of Reviews for {dailyNumOfReviewsEachRatings.columns[k]}",
                             line=dict(width=3)))

fig.update_layout(
    yaxis=dict(title_text="Number of Reviews", titlefont=dict(size=15)),
    xaxis=dict(title_text="Date", titlefont=dict(size=15)),
    title={'text': f"Number of Reviews Over Time for each Rating",
           'x': 0.5})
fig.update_traces(marker=dict(line=dict(color='#000000', width=0.75)))
fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})
fig.show()
'''''''''

# Rolling Mean of the Rating over Time (7 days window)
'''''''''
RatingDaily = data.sort_values('Time_submitted').resample('d', on='Time_submitted').Rating.mean().to_frame()
rollingRating = RatingDaily.rolling(7, min_periods=2).Rating.mean()

fig = go.Figure()
fig.add_trace(go.Scatter(y=rollingRating, x=rollingRating.index,
                         mode='markers+text+lines', name=f"Rolling Mean of the Ratings", line=dict(width=3)))

fig.update_layout(
    yaxis=dict(title_text="Average Rating", titlefont=dict(size=15)),
    xaxis=dict(title_text="Date", titlefont=dict(size=15)),
    title={'text': f"Rolling Mean of the Rating over Time (7 days window)",
           'x': 0.5})
fig.update_traces(marker=dict(line=dict(color='#000000', width=0.75)))
fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})
fig.show()
'''''''''

# Top used 100 Words before Text Cleaning
'''''''''
FreqOfWords = data['Review'].str.split(expand=True).stack().value_counts()
FreqOfWords_top100 = FreqOfWords[:100]

fig = px.treemap(FreqOfWords_top100, path=[FreqOfWords_top100.index], values=0)
fig.update_layout(title_text='Top used 100 Words before Text Cleaning',
                  title_x=0.5, title_font=dict(size=18)
                  )
fig.update_traces(textinfo="label+value")
fig.show()
'''''''''

# ----------------------------------------------------------------------------------------------------------------------

# Text Cleaning


# Lower case of the Reviews
data['Review'] = data['Review'].str.lower()


# ------------------------------

# Fix contractions such as doesn't to does not, he's to he is, etc.
# Check this link for detailed information: https://github.com/kootenpv/contractions
def fixContractions(inputs):
    return contractions.fix(inputs)


data['ReviewContractions'] = data['Review'].apply(fixContractions)

# ------------------------------

# Remove Numbers
data.ReviewContractions = data.ReviewContractions.replace(r'\d+', '', regex=True)


# Tokenization
def tokenization(inputs):
    return word_tokenize(inputs)


data['ReviewTokenized'] = data['ReviewContractions'].apply(tokenization)
print(data.ReviewTokenized[5])
print(data.ReviewTokenized[600])

# ------------------------------

# Stopwords Removal
stop_words = set(stopwords.words('english'))
stop_words.remove('not')


def stopwordsRemove(inputs):
    return [item for item in inputs if item not in stop_words]


data['ReviewStop'] = data['ReviewTokenized'].apply(stopwordsRemove)
print(data.ReviewStop[5])
print(data.ReviewStop[600])


# ------------------------------

# Remove punctuations from tokenized text rows
def removePunctuation(inputs):
    p = re.compile(r'[^\w\s]+')
    return p.sub('', inputs)


data.ReviewStop = data.ReviewStop.apply(lambda x: list(map(removePunctuation, x)))
print(data.ReviewStop[5])
print(data.ReviewStop[600])


# ------------------------------

# Removing Emojis from the text  # https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b


def removeEmoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)


data['ReviewStop'] = data['ReviewStop'].apply(lambda x: list(map(removeEmoji, x)))
print(data.ReviewStop[5])
print(data.ReviewStop[600])


# ------------------------------

# Remove non-latin words from the sentences. There are some chinese and russian words in the reviews
def removeNonLatin(sting):
    return re.sub(r'[^\x00-\x7f]', r'', sting)


data['ReviewStop'] = data['ReviewStop'].apply(lambda x: list(map(removeNonLatin, x)))


# Remove underscores
def removeUnderscores(sting):
    return str.replace(sting, '_', '')


data['ReviewStop'] = data['ReviewStop'].apply(lambda x: list(map(removeUnderscores, x)))


# ------------------------------


# Removing Words less than length 2. Because I replaced emojis and punctuations with ''.
# There are still in the text as a blank element of the sentence. After this step, they will be removed.
def removeLessThan_2(inputs):
    return [j for j in inputs if len(j) > 2]


data['ReviewStop'] = data['ReviewStop'].apply(removeLessThan_2)
print(data.ReviewStop[5])
print(data.ReviewStop[600])

# ------------------------------

# Lemmatization
lemmatizer = WordNetLemmatizer()


def lemmatization(inputs):
    return [lemmatizer.lemmatize(word=x, pos='v') for x in inputs]


data['ReviewLemmatized'] = data['ReviewStop'].apply(lemmatization)
print(data.ReviewLemmatized[5])
print(data.ReviewLemmatized[600])

# Stemming
# ps = PorterStemmer()
# data['ReviewLemmatized'] = data['ReviewLemmatized'] .apply(lambda x: [ps.stem(y) for y in x])
# print(data.ReviewLemmatized[5])
# print(data.ReviewLemmatized[600])

# ------------------------------

# Joining Tokens into Sentences
data['ReviewFinal'] = data['ReviewLemmatized'].str.join(' ')
print(data.ReviewFinal[5])
print(data.ReviewFinal[600])

# ----------------------------------------------------------------------------------------------------------------------

# Top used 100 Words after Text Cleaning
'''''''''
FreqOfWords = data['ReviewFinal'].str.split(expand=True).stack().value_counts()
FreqOfWords_top100 = FreqOfWords[:100]

fig = px.treemap(FreqOfWords_top100, path=[FreqOfWords_top100.index], values=0)
fig.update_layout(title_text='Top used 100 Words after Text Cleaning',
                  title_x=0.5, title_font=dict(size=18)
                  )
fig.update_traces(textinfo="label+value")
fig.show()
'''''''''

# Top used Words at Drop and Peak Dates of the Rating Rolling Average, what happened between this time periods?
# (12 April - 17 April) = pt1, (27 April - 7 May) = pt2 and (8 March - 14 March) = pt3
pt1_data = data[(data.Time_submitted >= '2022-04-12 00:00:00') & (data.Time_submitted <= '2022-04-17 23:59:59')]
pt2_data = data[(data.Time_submitted >= '2022-04-27 00:00:00') & (data.Time_submitted <= '2022-05-07 23:59:59')]
pt3_data = data[(data.Time_submitted >= '2022-03-08 00:00:00') & (data.Time_submitted <= '2022-03-14 23:59:59')]


# ------------------------------

# Create ngrams
def get_ngrams(text_input, n):  # https://stackoverflow.com/a/32307986
    n_grams = ngrams(word_tokenize(text_input), n)
    return [' '.join(grams) for grams in n_grams]


# ------------------------------

# Top 3grams between 12 April - 17 April (there was a huge drop in the Average Rating this range)
'''''''''
# ngrams of the 12 April - 17 April (huge drop)
pt1_ngrams = pd.DataFrame(pt1_data.ReviewFinal.apply(lambda x: get_ngrams(text_input=x, n=3)))
pt1_ngrams = pt1_ngrams.ReviewFinal.explode().value_counts().head(15)

fig = px.bar(x=pt1_ngrams.index, y=pt1_ngrams, color=pt1_ngrams, text=pt1_ngrams,
             color_continuous_scale=px.colors.sequential.OrRd)
fig.update_layout(xaxis={'categoryorder': 'total descending'})
fig.update_traces(marker=dict(line=dict(color='#000000', width=1)), textposition='outside')
fig.update_layout(
    yaxis=dict(title_text="Frequency", titlefont=dict(size=15)),
    xaxis=dict(title_text="ngrams", titlefont=dict(size=15)),
    title={'text': f"Top 3-grams between 12 April - 17 April (there was a huge drop in the Average Rating this range)",
           'x': 0.5})
fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})
fig.update_coloraxes(showscale=False)
fig.show()
'''''''''

# Top 3grams between 27 April - 7 May (there was a huge peak in the Average Rating this range)
'''''''''
# ngrams of the 27 April - 7 May (huge peak)
pt2_ngrams = pd.DataFrame(pt2_data.ReviewFinal.apply(lambda x: get_ngrams(text_input=x, n=3)))
pt2_ngrams = pt2_ngrams.ReviewFinal.explode().value_counts().head(15)

fig = px.bar(x=pt2_ngrams.index, y=pt2_ngrams, color=pt2_ngrams, text=pt2_ngrams,
             color_continuous_scale=px.colors.sequential.OrRd)
fig.update_layout(xaxis={'categoryorder': 'total descending'})
fig.update_traces(marker=dict(line=dict(color='#000000', width=1)), textposition='outside')
fig.update_layout(
    yaxis=dict(title_text="Frequency", titlefont=dict(size=15)),
    xaxis=dict(title_text="ngrams", titlefont=dict(size=15)),
    title={'text': f"Top 3-grams between 27 April - 7 May (there was a huge peak in the Average Rating this range)",
           'x': 0.5})
fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})
fig.update_coloraxes(showscale=False)
fig.show()
'''''''''

# Top 3grams between 8 March - 14 March (there was a drop peak in the Average Rating this range)
'''''''''
# ngrams of the 8 March - 14 March (huge drop)
pt3_ngrams = pd.DataFrame(pt3_data.ReviewFinal.apply(lambda x: get_ngrams(text_input=x, n=3)))
pt3_ngrams = pt3_ngrams.ReviewFinal.explode().value_counts().head(15)

fig = px.bar(x=pt3_ngrams.index, y=pt3_ngrams, color=pt3_ngrams, text=pt3_ngrams,
             color_continuous_scale=px.colors.sequential.OrRd)
fig.update_layout(xaxis={'categoryorder': 'total descending'})
fig.update_traces(marker=dict(line=dict(color='#000000', width=1)), textposition='outside')
fig.update_layout(
    yaxis=dict(title_text="Frequency", titlefont=dict(size=15)),
    xaxis=dict(title_text="ngrams", titlefont=dict(size=15)),
    title={'text': f"Top 3-grams between 8 March - 14 March (there was a drop peak in the Average Rating this range)",
           'x': 0.5})
fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})
fig.update_coloraxes(showscale=False)
fig.show()
'''''''''

# ------------------------------

# Correlation between the Length of Text and Total Thumbs Up
'''''''''
fig = px.scatter(data,
                 x="length_of_text",
                 y="Total_thumbsup",
                 trendline="ols")
fig.update_traces(marker=dict(line=dict(color='#000000', width=1)))
fig.update_layout(
    yaxis=dict(title_text="Total Thumbs Up", titlefont=dict(size=15)),
    xaxis=dict(title_text="Length of Text", titlefont=dict(size=15)),
    title={'text': f"Correlation between the Length of Text and Total Thumbs Up",
           'x': 0.5})
fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})
fig.data[1]['line'].color = 'red'
fig.show()
'''''''''

# Correlation Graph of the Dataset
'''''''''
fig = px.imshow(data.corr().round(2), text_auto=True, aspect="auto",
                color_continuous_scale='Reds')
fig.update_layout(
    title={'text': f"Correlation Graph of the Dataset",
           'x': 0.5})
fig.update_layout(coloraxis_showscale=False)
fig.show()
'''''''''

# ------------------------------

# Top 2grams for Reviews that got 5 points
'''''''''
rating5_ngrams = pd.DataFrame(data[data.Rating == 5].ReviewFinal.apply(lambda x: get_ngrams(text_input=x, n=2)))
rating5_ngrams = rating5_ngrams.ReviewFinal.explode().value_counts().head(15)

fig = px.bar(x=rating5_ngrams.index, y=rating5_ngrams, color=rating5_ngrams, text=rating5_ngrams,
             color_continuous_scale=px.colors.sequential.OrRd)
fig.update_layout(xaxis={'categoryorder': 'total descending'})
fig.update_traces(marker=dict(line=dict(color='#000000', width=1)), textposition='outside')
fig.update_layout(
    yaxis=dict(title_text="Frequency", titlefont=dict(size=15)),
    xaxis=dict(title_text="ngrams", titlefont=dict(size=15)),
    title={'text': f"Top 2grams for Reviews that got 5 Points",
           'x': 0.5})
fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})
fig.update_coloraxes(showscale=False)
fig.show()
'''''''''

# Top 3grams for Reviews that got 5 points
'''''''''
rating5_ngrams = pd.DataFrame(data[data.Rating == 5].ReviewFinal.apply(lambda x: get_ngrams(text_input=x, n=3)))
rating5_ngrams = rating5_ngrams.ReviewFinal.explode().value_counts().head(15)

fig = px.bar(x=rating5_ngrams.index, y=rating5_ngrams, color=rating5_ngrams, text=rating5_ngrams,
             color_continuous_scale=px.colors.sequential.OrRd)
fig.update_layout(xaxis={'categoryorder': 'total descending'})
fig.update_traces(marker=dict(line=dict(color='#000000', width=1)), textposition='outside')
fig.update_layout(
    yaxis=dict(title_text="Frequency", titlefont=dict(size=15)),
    xaxis=dict(title_text="ngrams", titlefont=dict(size=15)),
    title={'text': f"Top 3grams for Reviews that got 5 Points",
           'x': 0.5})
fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})
fig.update_coloraxes(showscale=False)
fig.show()
'''''''''

# ------------------------------

# Top 2grams for Reviews that got 1 or 2 Points
'''''''''
rating5_ngrams = pd.DataFrame(data[data.Rating <= 2].ReviewFinal.apply(lambda x: get_ngrams(text_input=x, n=2)))
rating5_ngrams = rating5_ngrams.ReviewFinal.explode().value_counts().head(15)

fig = px.bar(x=rating5_ngrams.index, y=rating5_ngrams, color=rating5_ngrams, text=rating5_ngrams,
             color_continuous_scale=px.colors.sequential.OrRd)
fig.update_layout(xaxis={'categoryorder': 'total descending'})
fig.update_traces(marker=dict(line=dict(color='#000000', width=1)), textposition='outside')
fig.update_layout(
    yaxis=dict(title_text="Frequency", titlefont=dict(size=15)),
    xaxis=dict(title_text="ngrams", titlefont=dict(size=15)),
    title={'text': f"Top 2grams for Reviews that got 1 or 2 Points",
           'x': 0.5})
fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})
fig.update_coloraxes(showscale=False)
fig.show()
'''''''''

# Top 3grams for Reviews that got 1 or 2 Points
'''''''''
rating5_ngrams = pd.DataFrame(data[data.Rating <= 2].ReviewFinal.apply(lambda x: get_ngrams(text_input=x, n=3)))
rating5_ngrams = rating5_ngrams.ReviewFinal.explode().value_counts().head(15)

fig = px.bar(x=rating5_ngrams.index, y=rating5_ngrams, color=rating5_ngrams, text=rating5_ngrams,
             color_continuous_scale=px.colors.sequential.OrRd)
fig.update_layout(xaxis={'categoryorder': 'total descending'})
fig.update_traces(marker=dict(line=dict(color='#000000', width=1)), textposition='outside')
fig.update_layout(
    yaxis=dict(title_text="Frequency", titlefont=dict(size=15)),
    xaxis=dict(title_text="ngrams", titlefont=dict(size=15)),
    title={'text': f"Top 3grams for Reviews that got 1 or 2 Points",
           'x': 0.5})
fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})
fig.update_coloraxes(showscale=False)
fig.show()
'''''''''

# ------------------------------

# Top 2grams for Spotify Replies
'''''''''
# Text Cleaning for Spotify's Replies to the Users
not_nan_replies = data[~data.Reply.isna()]
not_nan_replies['Reply'] = not_nan_replies['Reply'].str.lower()
not_nan_replies['Reply'] = not_nan_replies['Reply'].apply(fixContractions)
not_nan_replies['Reply'] = not_nan_replies['Reply'].apply(tokenization)
not_nan_replies['Reply'] = not_nan_replies['Reply'].apply(stopwordsRemove)
not_nan_replies.Reply = not_nan_replies.Reply.apply(lambda x: list(map(removePunctuation, x)))
not_nan_replies['Reply'] = not_nan_replies['Reply'].apply(lambda x: list(map(removeEmoji, x)))
not_nan_replies['Reply'] = not_nan_replies['Reply'].apply(removeLessThan_2)
not_nan_replies['Reply'] = not_nan_replies['Reply'].apply(lemmatization)
not_nan_replies['Reply'] = not_nan_replies['Reply'].str.join(' ')

reply_ngrams = pd.DataFrame(not_nan_replies.Reply.apply(lambda x: get_ngrams(text_input=x, n=2)))
reply_ngrams = reply_ngrams.Reply.explode().value_counts().head(15)

fig = px.bar(x=reply_ngrams.index, y=reply_ngrams, color=reply_ngrams, text=reply_ngrams,
             color_continuous_scale=px.colors.sequential.OrRd)
fig.update_layout(xaxis={'categoryorder': 'total descending'})
fig.update_traces(marker=dict(line=dict(color='#000000', width=1)), textposition='outside')
fig.update_layout(
    yaxis=dict(title_text="Frequency", titlefont=dict(size=15)),
    xaxis=dict(title_text="ngrams", titlefont=dict(size=15)),
    title={'text': f"Top 2grams for Spotify Replies",
           'x': 0.5})
fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})
fig.update_coloraxes(showscale=False)
fig.show()
'''''''''

# Top 3grams for Spotify Replies
'''''''''
reply_ngrams = pd.DataFrame(not_nan_replies.Reply.apply(lambda x: get_ngrams(text_input=x, n=3)))
reply_ngrams = reply_ngrams.Reply.explode().value_counts().head(15)

fig = px.bar(x=reply_ngrams.index, y=reply_ngrams, color=reply_ngrams, text=reply_ngrams,
             color_continuous_scale=px.colors.sequential.OrRd)
fig.update_layout(xaxis={'categoryorder': 'total descending'})
fig.update_traces(marker=dict(line=dict(color='#000000', width=1)), textposition='outside')
fig.update_layout(
    yaxis=dict(title_text="Frequency", titlefont=dict(size=15)),
    xaxis=dict(title_text="ngrams", titlefont=dict(size=15)),
    title={'text': f"Top 3grams for Spotify Replies",
           'x': 0.5})
fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})
fig.update_coloraxes(showscale=False)
fig.show()
'''''''''

# ----------------------------------------------------------------------------------------------------------------------

# Combining 5 different Labels into 2
data.Rating = data.Rating.replace({5: 'Good', 4: 'Good', 3: 'Bad', 2: 'Bad', 1: 'Bad'})

# ----------------------------------------------------------------------------------------------------------------------

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(data.ReviewFinal,
                                                    data.Rating,
                                                    test_size=0.2,
                                                    random_state=13)

print(f"X_train Shape:{X_train.shape}")
print(f"X_test Shape: {X_test.shape}")

# ----------------------------------------------------------------------------------------------------------------------

# Creating TF-IDF Model
tf_idf_model = TfidfVectorizer()

# Fitting the TF-IDF model and transforming data
tf_idf_vector_train = tf_idf_model.fit_transform(X_train)
tf_idf_vector_test = tf_idf_model.transform(X_test)

# ----------------------------------------------------------------------------------------------------------------------

# Model Creation: Random Forest Classifier
'''''''''
rf = RandomForestClassifier(max_depth=60, min_samples_leaf=5,
                            min_samples_split=6, n_estimators=60, oob_score=True,
                            max_leaf_nodes=55, random_state=13)
rf.fit(tf_idf_vector_train, y_train)
predictions_rf_test = pd.Series(rf.predict(tf_idf_vector_test))

train_scoreRF = rf.score(tf_idf_vector_train, y_train)
test_scoreRF = rf.score(tf_idf_vector_test, y_test)

# Train and Test R2 Scores
print(f"RF Train R2 Score: {round(train_scoreRF, 3)}")
print(f"RF Test R2 Score: {round(test_scoreRF, 3)}")

# Classification Report for Test Data
print(classification_report(y_test, predictions_rf_test))

# --------------------

# Cross-Validation of the Random Forest Classifier
cv_rf = cross_validate(rf, tf_idf_vector_test, y_test, cv=5)
print('CV Scores:', cv_rf['test_score'])
print('CV Scores Average: %', round(cv_rf['test_score'].mean() * 100, 3))
print('CV Scores Standard Deviation: %', round(cv_rf['test_score'].std() * 100, 3))
print(f'CV Scores Range: %{((cv_rf["test_score"].mean() + cv_rf["test_score"].std()) * 100).round(3)} - '
      f'%{((cv_rf["test_score"].mean() - cv_rf["test_score"].std()) * 100).round(3)}')

fig = go.Figure()
fig.add_trace(go.Scatter(y=cv_rf['test_score'],
                         text=cv_rf['test_score'],
                         mode='markers+text+lines',
                         name=f"Number of Reviews", line=dict(width=3)))
fig.add_hline(cv_rf['test_score'].mean(), line_width=2, line_dash="dash", line_color="red",
              annotation_text="Average Score", annotation_position="top right")
fig.update_layout(
    yaxis=dict(title_text="Scores", titlefont=dict(size=15)),
    xaxis=dict(title_text="Fold Number", titlefont=dict(size=15)),
    title={'text': f"K-Fold Cross Validation Scores (Random Forest Classifier)",
           'x': 0.5})
fig.update_traces(marker=dict(line=dict(color='#000000', width=0.75)), texttemplate='%{text:.3}',
                  textposition='top center')
fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})
fig.update_yaxes(range=[0.75, 0.85])
fig.show()
'''''''''

# ----------------------------------------------------------------------------------------------------------------------

# Model Creation: XGBClassifier
'''''''''
XGB = XGBClassifier(learning_rate=0.005, max_depth=10, n_estimators=30,
                    colsample_bytree=0.3, min_child_weight=0.5, reg_alpha=0.3, random_state=13)
XGB.fit(tf_idf_vector_train, y_train)

predictions_XGB_test = pd.Series(XGB.predict(tf_idf_vector_test))

train_scoreXGB = XGB.score(tf_idf_vector_train, y_train)
test_scoreXGB = XGB.score(tf_idf_vector_test, y_test)

# Train and Test R2 Scores
print(f"XGB Train R2 Score: {round(train_scoreXGB, 3)}")
print(f"XGB Test R2 Score: {round(test_scoreXGB, 3)}")

# Classification Report for Test Data
print(classification_report(y_test, predictions_XGB_test))

# --------------------

# Cross-Validation of the XGBClassifier
cv_XGB = cross_validate(XGB, tf_idf_vector_test, y_test, cv=5)
print('CV Scores:', cv_XGB['test_score'])
print('CV Scores Average: %', round(cv_XGB['test_score'].mean() * 100, 3))
print('CV Scores Standard Deviation: %', round(cv_XGB['test_score'].std() * 100, 3))
print(f'CV Scores Range: %{((cv_XGB["test_score"].mean() + cv_XGB["test_score"].std()) * 100).round(3)} - '
      f'%{((cv_XGB["test_score"].mean() - cv_XGB["test_score"].std()) * 100).round(3)}')

fig = go.Figure()
fig.add_trace(go.Scatter(y=cv_XGB['test_score'],
                         text=cv_XGB['test_score'],
                         mode='markers+text+lines',
                         name=f"Number of Reviews", line=dict(width=3)))
fig.add_hline(cv_XGB['test_score'].mean(), line_width=2, line_dash="dash", line_color="red",
              annotation_text="Average Score", annotation_position="top right")
fig.update_layout(
    yaxis=dict(title_text="Scores", titlefont=dict(size=15)),
    xaxis=dict(title_text="Fold Number", titlefont=dict(size=15)),
    title={'text': f"K-Fold Cross Validation Scores (XGBClassifier)",
           'x': 0.5})
fig.update_traces(marker=dict(line=dict(color='#000000', width=0.75)), texttemplate='%{text:.3}',
                  textposition='top center')
fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})
fig.update_yaxes(range=[0.75, 0.85])
fig.show()
'''''''''

# ----------------------------------------------------------------------------------------------------------------------

# Model Creation: Logistic Regression
'''''''''
lr = LogisticRegression(random_state=13, max_iter=2000).fit(tf_idf_vector_train, y_train)

predictions_LR_test = pd.Series(lr.predict(tf_idf_vector_test))

train_scoreLR = lr.score(tf_idf_vector_train, y_train)
test_scoreLR = lr.score(tf_idf_vector_test, y_test)

# Train and Test R2 Scores
print(f"LR Train R2 Score: {round(train_scoreLR, 3)}")
print(f"LR Test R2 Score: {round(test_scoreLR, 3)}")

# Classification Report for Test Data
print(classification_report(y_test, predictions_LR_test))

# --------------------

# Cross-Validation of the Logistic Regression
cv_LR = cross_validate(lr, tf_idf_vector_test, y_test, cv=5)
print('CV Scores:', cv_LR['test_score'])
print('CV Scores Average: %', round(cv_LR['test_score'].mean() * 100, 3))
print('CV Scores Standard Deviation: %', round(cv_LR['test_score'].std() * 100, 3))
print(f'CV Scores Range: %{((cv_LR["test_score"].mean() + cv_LR["test_score"].std()) * 100).round(3)} - '
      f'%{((cv_LR["test_score"].mean() - cv_LR["test_score"].std()) * 100).round(3)}')

fig = go.Figure()
fig.add_trace(go.Scatter(y=cv_LR['test_score'],
                         text=cv_LR['test_score'],
                         mode='markers+text+lines',
                         name=f"Number of Reviews", line=dict(width=3)))
fig.add_hline(cv_LR['test_score'].mean(), line_width=2, line_dash="dash", line_color="red",
              annotation_text="Average Score", annotation_position="top right")
fig.update_layout(
    yaxis=dict(title_text="Scores", titlefont=dict(size=15)),
    xaxis=dict(title_text="Fold Number", titlefont=dict(size=15)),
    title={'text': f"K-Fold Cross Validation Scores (Logistic Regression)",
           'x': 0.5})
fig.update_traces(marker=dict(line=dict(color='#000000', width=0.75)), texttemplate='%{text:.3}',
                  textposition='top center')
fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})
# fig.update_yaxes(range=[0.75, 0.85])
fig.show()
'''''''''

# ----------------------------------------------------------------------------------------------------------------------

# Model Creation: LightGBM
'''''''''
lgb = lgb.LGBMClassifier().fit(tf_idf_vector_train, y_train)

predictions_LGB_test = pd.Series(lgb.predict(tf_idf_vector_test))

train_scoreLGB = lgb.score(tf_idf_vector_train, y_train)
test_scoreLGB = lgb.score(tf_idf_vector_test, y_test)

# Train and Test R2 Scores
print(f"LGB Train R2 Score: {round(train_scoreLGB, 3)}")
print(f"LGB Test R2 Score: {round(test_scoreLGB, 3)}")

# Classification Report for Test Data
print(classification_report(y_test, predictions_LGB_test))

# --------------------

# Cross-Validation of the LightGBM
cv_LGB = cross_validate(lgb, tf_idf_vector_test, y_test, cv=5)
print('CV Scores:', cv_LGB['test_score'])
print('CV Scores Average: %', round(cv_LGB['test_score'].mean() * 100, 3))
print('CV Scores Standard Deviation: %', round(cv_LGB['test_score'].std() * 100, 3))
print(f'CV Scores Range: %{((cv_LGB["test_score"].mean() + cv_LGB["test_score"].std()) * 100).round(3)} - '
      f'%{((cv_LGB["test_score"].mean() - cv_LGB["test_score"].std()) * 100).round(3)}')

fig = go.Figure()
fig.add_trace(go.Scatter(y=cv_LGB['test_score'],
                         text=cv_LGB['test_score'],
                         mode='markers+text+lines',
                         name=f"Number of Reviews", line=dict(width=3)))
fig.add_hline(cv_LGB['test_score'].mean(), line_width=2, line_dash="dash", line_color="red",
              annotation_text="Average Score", annotation_position="top right")
fig.update_layout(
    yaxis=dict(title_text="Scores", titlefont=dict(size=15)),
    xaxis=dict(title_text="Fold Number", titlefont=dict(size=15)),
    title={'text': f"K-Fold Cross Validation Scores (LightGBM)",
           'x': 0.5})
fig.update_traces(marker=dict(line=dict(color='#000000', width=0.75)), texttemplate='%{text:.3}',
                  textposition='top center')
fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})
fig.update_yaxes(range=[0.84, 0.865])
fig.show()
'''''''''

# ----------------------------------------------------------------------------------------------------------------------

# Model Creation: Sequential Model (BiDirectional LSTM)

# Train Validation set Split
X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                  y_train,
                                                  test_size=0.25,
                                                  random_state=13)

print(f"X_train Shape:{X_train.shape}")
print(f"X_val Shape: {X_val.shape}")
print(f"X_test Shape: {X_test.shape}")

# --------------------

# Tokenizing with Tensorflow
num_words = 10000

# Define Tokenizer and fit it with the X_train
tokenizer = Tokenizer(num_words=num_words, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

# Apply the Tokenizer
Tokenized_train = tokenizer.texts_to_sequences(X_train)
Tokenized_val = tokenizer.texts_to_sequences(X_val)

print('Non-tokenized Version: ', X_train[0])
print('Tokenized Version: ', tokenizer.texts_to_sequences([X_train[0]]))
print('--' * 20)
print('Non-tokenized Version: ', X_train[54650])
print('Tokenized Version: ', tokenizer.texts_to_sequences([X_train[54650]]))

# --------------------

# Applying Padding
maxLen = 50
Padded_train = pad_sequences(Tokenized_train, maxlen=maxLen, padding='post')
Padded_val = pad_sequences(Tokenized_val, maxlen=maxLen, padding='post')

# --------------------

# Creating the Model
model = Sequential()

model.add(Embedding(num_words, 64, input_length=maxLen))
model.add(Dropout(0.75))

# model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, activation='relu', return_sequences=True)))
# model.add(Dropout(0.3))

model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8, activation='relu')))
model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))

model.summary()

# --------------------

label_ = {"Bad": 0, "Good": 1}
y_train = y_train.replace(label_)
y_val = y_val.replace(label_)
y_test = y_test.replace(label_)

# --------------------

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                  mode='auto', patience=5,
                                                  restore_best_weights=True)

epochs = 50
hist = model.fit(Padded_train, y_train, batch_size=64, epochs=epochs,
                 validation_data=(Padded_val, y_val),
                 callbacks=[early_stopping])

# --------------------

# Train and Validation Loss Graphs
fig = go.Figure()
fig.add_trace(go.Scatter(y=hist.history['loss'],
                         mode='lines',
                         name=f"Train Loss", line=dict(width=2)))

fig.add_trace(go.Scatter(y=hist.history['val_loss'],
                         mode='lines',
                         name=f"Validation Loss", line=dict(width=2)))

fig.update_layout(
    yaxis=dict(title_text="Loss", titlefont=dict(size=15)),
    xaxis=dict(title_text="Epochs", titlefont=dict(size=15)),
    title={'text': f"Train and Validation Loss Graphs",
           'x': 0.5})
fig.update_traces(marker=dict(line=dict(color='#000000', width=0.75)))
fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})
fig.show()

# ----------------------------------------------------------------------------------------------------------------------

# Preparing the Test Data
Tokenized_test = tokenizer.texts_to_sequences(X_test)
Padded_test = pad_sequences(Tokenized_test, maxlen=maxLen, padding='post')

# Evaluating the Test Data
model.evaluate(Padded_test, y_test)

# ----------------------------------------------------------------------------------------------------------------------

# Classification Report for Test Data
y_test_prediction = model.predict(Padded_test).argmax(axis=1)
print(classification_report(y_test, y_test_prediction))

# ----------------------------------------------------------------------------------------------------------------------

# Visualise the Word Embeddings
'''''''''
# Get the index-word dictionary
reverse_word_index = tokenizer.index_word

# Get the embedding layer from the model (i.e. first layer)
embedding_layer = model.layers[0]

# Get the weights of the embedding layer
embedding_weights = embedding_layer.get_weights()[0]

# Print the shape. Expected is (vocab_size, embedding_dim)
print(embedding_weights.shape)

import io

# Open writeable files
out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')

# Initialize the loop. Start counting at `1` because `0` is just for the padding
for word_num in range(1, maxLen):
    # Get the word associated at the current index
    word_name = reverse_word_index[word_num]

    # Get the embedding weights associated with the current index
    word_embedding = embedding_weights[word_num]

    # Write the word name
    out_m.write(word_name + "\n")

    # Write the word embedding
    out_v.write('\t'.join([str(x) for x in word_embedding]) + "\n")

# Close the files
out_v.close()
'''''''''

# ----------------------------------------------------------------------------------------------------------------------


# Trying KerasTuner
def create_model(hp):
    model_tuner = Sequential()

    model_tuner.add(Embedding(num_words, hp.Choice("output_dim", values=[8, 16, 32, 64]), input_length=maxLen))
    model_tuner.add(Dropout(0.75))

    model_tuner.add(tf.keras.layers.BatchNormalization())
    model_tuner.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hp.Choice('num_units', values=[8, 12, 16, 32]),
                                                                       activation='relu')))
    model_tuner.add(Dropout(0.5))

    model_tuner.add(Dense(2, activation='softmax'))

    model_tuner.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model_tuner


tuner = keras_tuner.BayesianOptimization(
    create_model,
    objective='val_loss',
    max_trials=15, overwrite=True)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                  mode='auto', patience=5,
                                                  restore_best_weights=True)

tuner.search(Padded_train, y_train, epochs=20, validation_data=(Padded_val, y_val), callbacks=[early_stopping])
best_model = tuner.get_best_models()[0]
