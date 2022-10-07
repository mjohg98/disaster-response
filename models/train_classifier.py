# import statements 
import sys 
import nltk
nltk.download(['punkt', 'wordnet', 'omw-1.4', 'stopwords'])

import pandas as pd 
import re 
import pickle 
from sqlalchemy import create_engine 
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report 
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline 
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier

stop_words = stopwords.words('english')

def load_data(database_filepath): 

    """
    Load data from SQLite database, split into predictor and response 
    variables 

    Args: 
        database_filepath (string): filepath of the SQLite database

    Returns 
        X: predictor variables (the messages)
        Y: response variables (the 36 categories)
        categories: list of category labels 

    """

    # load dataframe from sqlite database
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('table', engine)

    # define predictor and response variables 
    # (messages and categories respectively)
    X = df['message']
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    categories = list(y.columns)

    return X, y, categories 


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


def build_model(): 

    """
    Defines pipeline model to perform message classification

    1) Define parameter grid to search over
    2) Apply GridSearchCV to optimise random forest classifier 
    3) Construct pipeline; there are three steps 

        i) CountVectorizer: convert text into a matrix of token counts
        
        ii) TfidfTransformer: transform token matrix to a normalised td-idf 
        representation
        
        iii) MultiOutputClassifier: classify multiple categories using random 
        forest classifier with hyperparamters optimised by grid-search 
    
    Args: 
        none 

    Returns: 
        pipeline: classifier model to train 


    """

    # define parameters to search over 
    parameters = {
        'n_estimators': [200, 300, 400, 500], 
        'criterion' : ['gini', 'entropy'],
        'max_features': ['sqrt', 'log2']
        }
    
    # apply gridsearch to random forest classifier
    cv = GridSearchCV(estimator=RandomForestClassifier(), param_grid=parameters)

    # construct pipeline and adapt random forest to multi-output classifier 
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)), 
        ('tfidf', TfidfTransformer()), 
        ('clf', MultiOutputClassifier(estimator=cv))
    ])
    
    return pipeline 


def evaluate_model(model, X_test, Y_test, category_names): 

    """
    Predict class labels on test data and evaluates model performance 
    by printing precision, recall and f1-score for each category

    Args: 
        model: classifier model used to predict classes of messages
        X_test: message test data 
        Y_test: category response test data 
        category_names: a list of the category labels 

    Returns: 
        model: classifier model used to predict classes of messages
    
    """

    # predicted categories for test messages 
    y_pred = model.predict(X_test)
    
    # print classification report for each category
    for ii in range(len(category_names)): 
        
        category = category_names[ii]

        result = classification_report(Y_test.loc[:, [category]], y_pred[:, ii], 
        zero_division=1)
        
        print(result)    

    return model 


def save_model(model, model_filepath): 

    """
    Saves model as pickle file 

    Args: 
        model: model used to classify messages
        model_filepath: filepath to save model as pickle file
    
    """
    
    # save model as pickle file 
    with open(f'{model_filepath}', 'wb') as handle: 
        pickle.dump(model, handle)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()




