# Disaster Response Message Classification

## Running the Web Application 

1. Navigate to the project's directory. 
2. Run the following commands to create SQLlite database and classifier: 
 * `python3 data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
 * `python3 models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
3. Go to the `app` folder. 
4. Run `python3 run.py` to run the web application. 
5. Open web browser and go to the specified local host address (port 3000) which appears in terminal. 

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

### Homepage: 

![Web App Homepage 1](/images/homepage1.png)

![Web App Homepage 2](/images/homepage2.png)

### Results page: 

![Web App Results 1](/images/result1.png)

![Web App Results 2](/images/result2.png)



