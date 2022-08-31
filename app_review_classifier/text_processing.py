import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.word2vec import Word2Vec
from gensim.models.phrases import Phrases, Phraser
from gensim.utils import simple_preprocess
import spacy

# Custom stop words
my_stop_words = ({'namely','between','up','whither','them','beside','your','about','hence','former','ours','itself','or','these','their','those','has','re','next','hereupon','whether','latter','towards','over','yourselves','himself','beforehand','you','wherever','another','than','do','around','him','upon','been','an','me','toward','within','of','whole','ca','once','nor','thru','seeming','already','keep','so','mine','others','until','move','ourselves','other','where','thereupon','she','am','without','again','hereby','be','someone','sometime','used','go','everyone','some','then','see','to','seemed','i','become','whatever','and','what','that','thence','too','whenever','whereupon','can','his','just','due','thereby','done','name','none','part','noone','since','doing','meanwhile','via','herself','also','amount','seems','say','get','through','show','made','such','a','as','hers','the','my','whereafter','in','themselves','which','something','put','it','by','may','who','various','whence','throughout','during','hereafter','had','after','under','few','using','whom','will','though','its','might','across','most','above','how','regarding','being','our','afterwards','behind','make','almost','each','side','along','much','while','any','elsewhere','many','this','own','us','would','does','latterly','anything','when','are','with','onto','even','did','thereafter','yours','all','is','third','if','somewhere','nothing','because','wherein','whoever','somehow','either','every','out','whose','front','take','both','they','for','empty','anyone','back','formerly','whereby','full','here','into','myself','we','became','from','seem','anywhere','besides','herein','ever','at','her','must','therein','nobody','well','give','per','indeed','down','still','on','could','although','amongst','there','was','else','first','further','have','several','yourself','beyond','now'}
                 | {'ill','i','id','ive','im','mine', 'you','youll','youre','your','youd','youve','yours', 'he','hell','hes','hed','his','she','shell','shed','hers','they','theyre','theyd','their','theirs','theyve','weve','wed','well','our','ours','isnt','wont','shant','d','x'}) # My additional ones since I got rid of apostrophes


# Default embedder configs
default_configs = {'textcleaner':{'stop_words':True,
                                  'ngrams':True,
                                  'lemmatization':True},
    
                   'tfidf':{'min_df':5, # Min number of occurances of word to consider
                            'ngram_range':(1, 1)}, # Ngram range
                   
                   'word2vec':{'min_count':5, # Min number of occurances of word to consider
                               'vector_size':50, # Dimension of embedding matrix (i.e. number of embedding features)
                               'window':5, # Size of window to consider
                               'workers':4, # Number of partitions during training
                               'sg':1}} # Training algorothm (CBOW(0) or skip gram(1))



class TextCleaner:
    def __init__(self, 
                 config=None):
        
        self.bigram_model = None
        self.trigram_model = None
        
        # Set config params
        if config is None:
            self.config = default_configs['textcleaner']
        else:
            self.config = config
        self.use_stop_words = self.config['stop_words']
        self.use_ngrams = self.config['ngrams']
        self.use_lemmatization = self.config['lemmatization']
        self.stop_words = my_stop_words
        
    def remove_stopwords_row(self, text, min_n=3):
        '''
        Removes stop words from single row (e.g. 'here is some text')
        '''
        txt = [token for token in text if token not in self.stop_words]
        # return None if not enough words left
        if len(txt) >= min_n:
            return ' '.join(txt)
        else:
            return None
    
    
    def remove_stopwords(self, text_df):

        '''
        Cleans whole text column and removes stop words e.g. df['cleaned_text'] = remove_stopwords(df['raw_text'])
        '''
        print('Cleaning up text and removing stopwords...')
        t1 = time.time()
        cleaned_text = [re.sub("[^a-zA-Z ]", '', str(row)).lower() for row in text_df] # Doesn't do anything for this dataset (remove non alpha-numeric?)
        cleaned_text = [re.sub(' +', ' ', str(row)).lower().lstrip(' ') for row in cleaned_text] # Remove multiple spaces
        cleaned_text = [self.remove_stopwords_row(doc.split(' ')) for doc in cleaned_text] # Remove stop words
        print('- Text cleaned in: {} seconds'.format(round((time.time() - t1), 2)))

        return cleaned_text

    
    def lemmatization(self, documents, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):

        # Load model
        print('\nLoading spacy model...')
        t1 = time.time()
        nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        print('- Model loaded in: {} seconds'.format(round((time.time() - t1), 2)))
        
        # Lemmatize text
        print('\nLemmatizing...')
        t1 = time.time()

        documents_out = []
        for document in documents:
            doc = nlp(" ".join(document)) 
            documents_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        
        print('- Lemmatizing done in: {} seconds'.format(round((time.time() - t1), 2)))
        return documents_out
        
        
    def train_ngrams(self, documents, ngram_threshold):
        # Build the bigram and trigram models
        print('\nTraining ngrams...')
        t1 = time.time()

        bigram = Phrases(documents, min_count=1, threshold=ngram_threshold) # higher threshold fewer phrases.
        trigram = Phrases(bigram[documents], min_count=1, threshold=ngram_threshold)  
        self.bigram_model = Phraser(bigram)
        self.trigram_model = Phraser(trigram)

        print('- Training done in: {} seconds'.format(round((time.time() - t1), 2)))


    def process_raw_text(self, raw_text, 
                               train_ngrams = False, 
                               ngram_threshold=200):

        def sentence_to_words(sentences):
            for sentence in sentences:
                yield(simple_preprocess(str(sentence), deacc=True, min_len=1, max_len=20))  # deacc=True removes punctuations

        # Remove stop words (if required)
        if self.use_stop_words == True:
            cleaned_text = self.remove_stopwords(raw_text)
        else:
            cleaned_text = raw_text

        # Turn to list of words and remove punctuation
        words = list(sentence_to_words(cleaned_text))

        # Retrain ngrams (if required)
        if train_ngrams == True:
            self.train_ngrams(words, ngram_threshold)
            
        # Apply ngrams (if required)
        if self.use_ngrams == True:
            if self.bigram_model is None:
                raise ValueError('Bigram/Trigram models not trained yet! Set train_ngrams=True when calling process_raw_text() to train and save the ngram models')
            else:
                words = [self.trigram_model[self.bigram_model[doc]] for doc in words]
        
        # Lemmatise (if required)
        if self.use_lemmatization == True:
            final_words = self.lemmatization(words)
        else:
            final_words = words

        return final_words
        
        

class Embedder:
    '''
    Class for performing embedding on cleaned text. Can use TfidfVectorizer or word2vec
    '''
    def __init__(self,
                 method='tfidf',
                 config = None):

        # Check method exists
        possible_methods = ['tfidf','word2vec']
        if method not in possible_methods:
            raise ValueError('Method not recognised. try one of: '+str(possible_methods))
        else:
            self.method = method
            
        # Set config params
        if config is None:
            self.config = default_configs[method]
        else:
            self.config = config
            
        # Set up models
        if self.method is 'tfidf':
            self.embedder = TfidfVectorizer(min_df=self.config['min_df'], 
                                            ngram_range=self.config['ngram_range'])
        elif self.method is 'word2vec':
            self.embedder = None # Gets created when training
        elif self.method is 'glove':
            print('not done yet')
        
        
    def fit(self, documents):
        
        if self.method is 'tfidf':
            self.embedder.fit([' '.join(document) for document in documents]) # Requires single string with spaces
            
        elif self.method is 'word2vec':
            print('Fitting word2vec model...')
            print('- Setting up model...')
            self.embedder = Word2Vec(min_count=self.config['min_count'],
                                     vector_size = self.config['vector_size'],
                                     workers = self.config['workers'],
                                     window = self.config['window'],
                                     sg = self.config['sg'])
            print('- Done!\n- Building Vocab...')
            self.embedder.build_vocab(documents, progress_per=1000)
            print('- Built!\n- Training Model...')
            self.embedder.train(documents, total_examples=self.embedder.corpus_count, epochs=50, report_delay=1)
            print('- Trained!')
            
            

    def apply(self, documents):
        
        if self.method is 'tfidf':
            return self.embedder.transform([' '.join(document) for document in documents]) # Requires single string with spaces
        
        elif self.method is 'word2vec':
            
            words = set(test_embedder.embedder.wv.index_to_key)

            # Get vectors of each word
            word_vectors = np.array([np.array([self.embedder.wv[i] for i in ls if i in words])
                                     for ls in documents], dtype=object)
            # Average this for all sentences
            document_vectors = []
            for v in word_vectors:
                if v.size > 1:
                    document_vectors.append(v.mean(axis=0))
                else:
                    document_vectors.append(np.zeros(self.config['vector_size'], dtype=float))


            return np.array(document_vectors)
