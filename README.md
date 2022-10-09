# Disaster Response Message Classification

The web application home page displays a few visuals summarising aspects of the dataset and an interface for viewers to classify their own messages using the model described below. 

## Data

The dataset consisted of 26,248 messages that were sent in the midst of natural disasters. Each message has been labelled a 0 or 1 ("Yes"/"No") with respect to 36 categories of messages. 

## Classification Model

The model used was constructed as an sklearn pipeline and consists of the following steps: 

* CountVectorizer - converts a collection of text documents to a matrix of token counts.
* TfidfTransformer - transforms the matrix of token counts to a tf-idf (term frequency-inverse document frequency) representation.
* MultiOutputClassifier - using a random forest as the base classifier, predicts labels for the 36 categories. 

## Files 

* The `app` folder contains the following files: 

  * `run.py` is the script that runs the web application, renders the home and results pages from the `go.html` and `master.html` templates in the `templates` subfolder. 

* The `data` folder contains: 

  * `disaster_messages.csv` - a .csv file containing the disaster messages. 
  * `diasster_categories.csv` - a .csv file containing the labels of each message. 
  * `process_data.py` - this script imports and merges data from `disaster_messages.csv` and `disaster_categories.csv`, cleans the data and saves it into an SQLlite database. 

* The `models` folder contains: 

  * `train_classifier.py` - this script loads the data from the SQLlite database and splits the data into a training and test sets. Then the classifier is created, trained using the training set and evaulated on the test set. 

## Screenshots of web application



