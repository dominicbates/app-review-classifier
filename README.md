# app-review-classifier

Module for processing and classifying text. Contains 3 scripts:

- **review-scraper.py**: For downloading review from the apple and google play stores
- **text-processing.py**: For cleaning text (removing stop words, getting ngrams, lemmatizing), and getting embeddings (TFIDF, Word2Vec)
- **classifier.py**: For training classifiers from these embeddings, or producing topic models (not finished yet)
