# import statements 
import json
import plotly
import pandas as pd
import re 
import numpy as np 

import nltk 
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Histogram 
import joblib
from sqlalchemy import create_engine

app = Flask(__name__)

# define stopwords 
stop_words = stopwords.words('english')

def tokenize(text): 

    """
    Tokenize a string into lemmatised words

    1) Remove punctuation from text
    2) Remove stopwords 
    3) Lemmatise text and normalise to lowercase, no whitespaces 

    Args: 
        text (string): the text to tokenise

    Returns 
        clean_tokens: a list of words of tokenised text 
    
    """

    # remove punctuation and tokenise text 
    tokens = word_tokenize(re.sub(r"[^a-zA-Z0-9]", " ", text))
    
    # remove stopwords 
    no_stopwords = [w for w in tokens if w not in stop_words]

    # initialise lemmatizsr 
    lemmatizer = WordNetLemmatizer()

    # return list of processed tokens 
    clean_tokens = []
    for tok in no_stopwords: 

        # lemmatise and normalise text
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens 


# load data
#folder = '/Users/michaelgerrard/Documents/Python/udacity/project2'
engine = create_engine(f'sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('table', engine)

# load model
model = joblib.load(f"../models/classifier.pkl")

# index webpage displays visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    """
    Extract data and create plotly visualisations to display on webpage 
    
    """
    
    facecolors = ['rgba(200, 0, 0, 0.5)', 'rgba(0, 200, 0, 0.5)', 
    'rgba(0, 0, 200, 0.5)']
    edgecolors = ['rgba(200, 0, 0, 0.8)', 'rgba(0, 200, 0, 0.8)', 
    'rgba(0, 0, 200, 0.8)']
    genres = ['direct', 'news', 'social']

    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # number of words per message by genre 
    group_df = df.groupby('genre')
    words = {}
    for ii in range(3): 

        # get dataframe for each genre
        group = group_df.get_group(genres[ii])

        # get word count of messages for each genre
        word_counts = group['message'].apply(lambda s: len(s.split()))
        word_counts = word_counts[word_counts <= 110]

        # save word counts in dictionary, with genre as key 
        words[genres[ii]] = word_counts 

    # number of messages per category, take top 10
    # breakdown categories by genre  
    categories = df.iloc[:, 4:]
    categories = categories[categories['related']!=2]

    # top 5 categories by number of messages 
    category_counts = categories.sum().sort_values(ascending=False)[:5]
    
    # compute number of messages of each genre, for each category
    top_5 = list(category_counts.index)
    top_5_genres = np.zeros((5, 3))
    for ii in range(len(top_5)): 
        x = df[df[top_5[ii]]==1]
        genre_sort = x['genre'].value_counts().sort_index()
        top_5_genres[ii, :] = genre_sort 

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts, 
                    marker=dict(
                        color=['rgba(200, 0, 0, 0.5)', 'rgba(0, 200, 0, 0.5)', 
                        'rgba(0, 0, 200, 0.5)'], 
                        line=dict(color=['rgba(200, 0, 0, 0.8)', 
                        'rgba(0, 200, 0, 0.8)', 'rgba(0, 0, 200, 0.8)'], width=2))
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },  
        {
            'data': [
                Bar(name=genres[ii],y=top_5_genres[:, ii], x=top_5,
                marker=dict(
                    color=facecolors[ii], 
                    line=dict(color=edgecolors[ii], width=2)),
                orientation='v'
                )
            for ii in range(3)],

            'layout': {
                'title': 'Top 5 Message Categories by Genre', 
                'yaxis': {'title': 'Number of Messages'},
                'barmode': 'stack'
            }

        }, 
        {   
            'data': [
                Histogram(
                    x=words[genre], 
                    marker=dict(
                        color=facecolors[ii],
                        line=dict(color=edgecolors[ii], width=1)), 
                    opacity=0.75, 
                    name=genre
                )
            for (ii, genre) in enumerate(genres)],
            
            'layout': {
                'title': 'Distribution of Word Counts by Genre',
                'yaxis': {'title': "Number of messages"},
                'xaxis': {'title': "Word count"},
                'barmode': 'overlay'
            }   
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()