# Sentiment-Analysis-through-Naive-Bayes-using-Bag-of-Words-implementation
A very well-known model in NLP is the Bag of Words model. It is a model used to preprocess the texts to classify before fitting the classification algorithms on the observations containing the texts.

## Install and Import the right libraries
  We will be needing numpy, and pandas to work with our dataset, and matplotlib for visualising it. We will be needing regex and nltk. You can install them by using the following commands on your console
  ```
  pip install regex
  pip install nltk
  ```
  
## Cleaning the texts
  We will be performing some basic text preprocessing like removing symbols, extra spaces and then use the PorterStemmer module from NLTK library to perform stemming. Also we will exclude certain "stopwords" from our dataset. 
  
## Create a bag of words
  ![bag_of_words](https://user-images.githubusercontent.com/55653469/83440362-70890680-a462-11ea-9fbb-e86e31d54337.png)

  A bag-of-words model, or BoW for short, is a way of extracting features from text for use in modeling, such as with machine learning algorithms.

The approach is very simple and flexible, and can be used in a myriad of ways for extracting features from documents.
A bag-of-words is a representation of text that describes the occurrence of words within a document. It involves two things:
* A vocabulary of known words.
* A measure of the presence of known words.
It is called a “bag” of words, because any information about the order or structure of words in the document is discarded. The model is only concerned with whether known words occur in the document, not where in the document.
## Splitting the Dataset
  Well, unlike any other model, we will need to split our dataset here as well.

## Train the Naive Bayes classifier
  Now we will train our model and predict the test results for our validation accuracy. We can improve the accuracy of our model by preprocessing in a more rigorous way. 
  
  I have uploaded two folders, in one I have implemented Naive Bayes' from scratch whereas in the other one, I have used the libraries. You can have a look at the code which I've implemented from scratch to understand the mathematics behind Naive Bayes' classification.
  For a more detailed understanding of the concept, go to: https://towardsdatascience.com/unfolding-na%C3%AFve-bayes-from-scratch-2e86dcae4b01
