# app-review-classifier


## Summary

Module for processing and classifying text. Contains 3 scripts:

- `review-scraper.py`: For downloading review from the apple and google play stores
- `text-processing.py`: For cleaning text (removing stop words, getting ngrams, lemmatizing), and getting embeddings (TFIDF, Word2Vec)
- `classifier.py`: For training classifiers from these embeddings, or producing topic models (not finished yet)



## Usage

### Scraping reviews

`review-scraper.get_all_reviews()` can be used to extract all reviews from both the apple and google play stores and combine these in to a single pandas `DataFrame`. However, due to the apple review scraper module sending HTTP requests in order to extract the reviews, requests will sometimes fail, meaning `get_all_reviews()` might be missing some reviews (despite some protection against this in the code)

For this reason you should instead keep a "production" file of reviews, and "update" this file each time you want to get the latest reviews. You can do this with `review-scraper.update_reviews_df(production_df)`, which will take an existing dataframe, attempt to extract all reviews, and then append any which are not already present in the existing dataframe.  This way, you should only ever be adding reviews, even if the request fails.

### Manual classification

A set of reviews have been manually classified, stored in this google sheet: (NOT YET CREATED). This is the dataset from which the classification models are trained. Reviews were extracted as above, which were filtered to (DATE RANGE), and randomised for manual classificaiton

### Data pre-processing

The `text-processing.TextCleaner` class can be used to remove stop words, generate bigrams and trigrams, and also lemmatize the text. This text can then be fed to the `text-processing.Embedder` class, which can produce either a tfidf vector, or word2vec embedding. Both of these 

### Models

The output of the embedder can easily be fed to a `classifier.Classifier` object, which can train a number of different supervised classification methods (e.g. LogisticRegression, MultinomialNB, DecisionTreeClassifier, xgboost...), to classify reviews.

### Production process

The production process would be:
- use `update_reviews_df()` to extract latest reviews and update the dataset
- Clean text with a (pretrained) `TextCleaner` object
- Embed text with a (pretrained) `Embedder` object
- Apply a (pretrained) `Classifier` object to produce final output
- Push this table to snowflake
