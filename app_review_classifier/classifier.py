import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import gensim.corpora as corpora
from xgboost import XGBClassifier


default_configs = {'lda':{'num_topics':8,
                          'passes':300, #2000, # Default 1 just tried 50
                          'iterations':100, # Default 50
                          'random_state':1,
                          'eval_every':10},

                    'XGBoost':{'max_depth':6,
                               'eta':0.3,
                               'min_child_weight':1,
                               'subsample':1,
                               'lambda':1,
                               'num_parallel_tree':1},

                    'LogisticRegression':None,
                    'MultinomialNB':None,
                    'DecisionTreeClassifier':None
                    }



class Classifier:
    def __init__(self, 
                 classifier='LogisticRegression',
                 config=None):
        
        # Check input first
        classifiers_list = ['LogisticRegression',
                            'MultinomialNB',
                            'DecisionTreeClassifier',
                            'XGBoost'] # No 'LinearSVC' because no predict_proba()
        if classifier not in classifiers_list:
            raise ValueError('Classifier "'+str(classifier)+'" not in list. Try one of:',classifiers_list)
        
        # Set config params
        if config is None:
            self.config = default_configs[classifier]
        else:
            self.config = config


        # Get relevant model
        self.classifier = classifier
        if classifier == 'LogisticRegression':
            self.model = LogisticRegression(C=2, penalty = 'l2', max_iter=1000, random_state=0)
        elif classifier == 'MultinomialNB':
            self.model = MultinomialNB()
        elif classifier == 'DecisionTreeClassifier':
            self.model = DecisionTreeClassifier(random_state=0)
        elif classifier == 'XGBoost':
            self.model = XGBClassifier(max_depth=self.config['max_depth'],
                                       eta=self.config['eta'],
                                       min_child_weight=config['min_child_weight'],
                                       subsample=config['subsample'],
                                       reg_lambda=config['lambda'],
                                       num_parallel_tree=config['num_parallel_tree'])
        # self.vectorizer = None
        self.model_fit = False
            

            
    def fit(self, vals_X, vals_y):
        self.model.fit(vals_X, vals_y)
        pred_y = self.apply(vals_X)
        self.print_performance(vals_y, pred_y)
        
        
    def apply(self, vals_X):
        if self.model_fit is None:
            raise ValueError('Model not trained yet. Run fit() first')
        else:
            return self.model.predict(vals_X)

    
    def apply_proba(self, vals_X):
        if self.model_fit is None:
            raise ValueError('Model not trained yet. Run fit() first')
        else:
            return self.model.predict_proba(vals_X)
    
    
    def print_performance(self, vals_y, pred_y, print_output=True):
        confusion_matrix_df = pd.DataFrame(data=confusion_matrix(vals_y, pred_y), columns=self.model.classes_, index=self.model.classes_)  
        accuracy_score_val = accuracy_score(vals_y, pred_y)

        if print==True:
            print('\nConfusion matrix:')
            print(confusion_matrix_df) 
            
            print('\nAccuracy:')
            print(accuracy_score_val)
        else:
            return confusion_matrix_df, accuracy_score_val
        
        
    # def print_top_words(self, vectorizer, n = 100):

    #     for c in range(0,len(self.model.classes_)):
    #         top_ind = np.argsort(self.model.coef_[c])[-1*n:][::-1]
    #         top_vals = ([self.model.coef_[c][i] for i in top_ind])

    #         inverse_vocabulary = dict((v,k) for k,v in vectorizer.vocabulary_.items())
    #         print('\nTop words for class:',self.model.classes_[c])
    #         words = [inverse_vocabulary[n]+', ' for n in top_ind]
    #         print(''.join(words)[:-2])
    #         print('')







class TopicModel:
    
    def __init__(self, 
                 config=None):
        
        # Set config params
        if config is None:
            self.config = default_configs['lda']
        else:
            self.config = config
            
        self.model = None
        self.num_topics = self.config['num_topics']
        self.training_passes = self.config['passes']
        self.training_iterations = self.config['iterations']
        self.training_random_state = self.config['random_state']
        self.training_eval_every = self.config['eval_every']

    
    def get_corpus(self, documents, train_dict=False):
        print('Getting corpus...')
        t1=time.time()
        if train_dict==True:
            print('- Training dictionary...')
            self.id2word = corpora.Dictionary(documents)
            
            if self.id2word is None:
                raise ValueError('No dictionary trained yet. Try ruynning get_corpus() with train=True')
            
        print('- Done in: {} seconds'.format(round((time.time() - t1), 2)))
        return [self.id2word.doc2bow(text) for text in cleaned_reviews]
                    

    def fit(self, documents):

        corpus = self.get_corpus(documents, train_dict=True)
        
        print('\nTraining model...')
        t1=time.time()
        self.model = LdaMulticore(corpus=corpus,
                                  id2word=self.id2word,
                                  num_topics=self.num_topics,
                                  passes = self.training_passes, # Default 1 just tried 50
                                  iterations = self.training_iterations, # Default 50
                                  random_state=self.training_random_state,
                                  eval_every=self.training_eval_every)        
        print('- Model trained in: {} seconds'.format(round((time.time() - t1), 2)))


    def apply(self, documents):
        corpus = self.get_corpus(documents, train_dict=False)





def split_sample(df, frac_train = 0.9, frac_validate = 0.1):
    '''
    Split dataframe in to a trianing, validation and test sample with these defined 
    fractions. Note 'frac_test' isn't needed since others should sum to 1
    '''
  
    # Shuffle dataframe
    df_shuffled = df.sample(frac=1, random_state=123).reset_index(drop=True)
    
    # Indexes
    train_index = int(len(df)*frac_train)
    validate_index = int(len(df)*(frac_train+frac_validate))

    # Sample dataframe
    df_train = df_shuffled[:train_index]
    df_validate = df_shuffled[train_index:validate_index]
    df_test = df_shuffled[validate_index:]
    
    return df_train, df_validate, df_test



