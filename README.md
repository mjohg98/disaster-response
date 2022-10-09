# Disaster Response Message Classification

## Data

The dataset consisted of 26,248 messages that were sent in the midst of natural disasters. Each message has been labelled a 0 or 1 ("Yes"/"No") with respect to 36 categories of messages. 

## Classification Model

The model used was constructed as an sklearn pipeline and consists of the following steps: 

* CountVectorizer - converts a collection of text documents to a matrix of token counts.
* TfidfTransformer - transforms the matrix of token counts to a tf-idf (term frequency-inverse document frequency) representation.
* MultiOutputClassifier - using a random forest as the base classifier, predicts labels for the 36 categories. 

### Results

## Files 

* `app`
