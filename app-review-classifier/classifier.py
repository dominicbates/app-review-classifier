import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score



class Classifier:
    def __init__(self, 
                 classifier='LogisticRegression'):
        
        # Check input first
        classifiers_list = ['LogisticRegression',
                            'MultinomialNB',
                            'DecisionTreeClassifier'] # No 'LinearSVC' this because no predict_proba()
        if classifier not in classifiers_list:
            raise ValueError('Classifier "'+str(classifier)+'" not in list. Try one of:',classifiers_list)
        
        # Get relevant model
        self.classifier = classifier
        if classifier == 'LogisticRegression':
            self.model = LogisticRegression(C=2, penalty = 'l2', max_iter=1000, random_state=0)
        elif classifier == 'MultinomialNB':
            self.model = MultinomialNB()
        elif classifier == 'DecisionTreeClassifier':
            self.model = DecisionTreeClassifier(random_state=0)
        self.vectorizer = None
        self.model_fit = False
            
    def fit_vectorizer(self, raw_text):
        self.vectorizer = TfidfVectorizer(min_df=5, ngram_range=(1, 1)) # Arbitary params try others? bigrams? ngram_range=(1, 2)
        self.vectorizer.fit(raw_text)
        
        
    def apply_vectorizer(self, raw_text):
        if self.vectorizer is None:
            raise ValueError('Vectorizer not trained yet. Run fit_vectorizer() on raw text')
        else:
            return self.vectorizer.transform(raw_text)
            
    def fit_model(self, vals_X, vals_y):
        self.model.fit(vals_X, vals_y)
        pred_y = self.apply_model(vals_X)
        self.print_confusion_matrix(vals_y, pred_y)
        
        
    def apply_model(self, vals_X):
        if self.model_fit is None:
            raise ValueError('Model not trained yet. Run fit_model() first')
        else:
            return self.model.predict(vals_X)

    
    def apply_model_proba(self, vals_X):
        if self.model_fit is None:
            raise ValueError('Model not trained yet. Run fit_model() first')
        else:
            return self.model.predict_proba(vals_X)
    
    
    def print_confusion_matrix(self, vals_y, pred_y):
        print('Confusion matrix:')
        print(confusion_matrix(vals_y, pred_y))
        
        print('\nAccuracy:')
        print(accuracy_score(vals_y, pred_y))
        
        
    def print_top_words(self, vectorizer, n = 100):

        for c in range(0,len(self.model.classes_)):
            top_ind = np.argsort(self.model.coef_[c])[-1*n:][::-1]
            top_vals = ([self.model.coef_[c][i] for i in top_ind])

            inverse_vocabulary = dict((v,k) for k,v in vectorizer.vocabulary_.items())
            print('Top words for class:',self.model.classes_[c])
            words = [inverse_vocabulary[n]+', ' for n in top_ind]
            print(''.join(words)[:-2])
            print('')




class TopicModel:
    def __init__(self):






