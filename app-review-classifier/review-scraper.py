import pandas as pd
import numpy as np
from app_store_scraper import AppStore
from google_play_scraper import reviews_all, reviews, Sort, app
import time


def get_apple_reviews(app_name, app_id, country):
    
    # Try and get scrape reviews...
    for n in range(20):
        reviews = AppStore(country = country, app_name = app_name, app_id = app_id)
        reviews.review(how_many=100000)
        reviews_dict = reviews.reviews
        del reviews
        
        # If no results, retry...
        if len(reviews_dict) < 2:
            print("No results found - let's wait a sec, then retry...")
            time.sleep(10)
        else:
            print('Found some results! Moving on...')
            break
            
    return reviews_dict



def apple_reviews_to_df(reviews_dict, app_name):
        # Turn to dataframe
    col_date = []
    col_review = []
    col_title = []
    col_rating = []
    col_app = []
    for review in reviews_dict:
        col_date.append(review['date'])
        col_title.append(review['title'])
        col_review.append(review['review'])
        col_rating.append(review['rating'])
        col_app.append(app_name)
    
    reviews_df = pd.DataFrame({'date':col_date,
                               'title':col_title,
                               'review':col_review,
                               'rating':col_rating,
                               'app':col_app})
    return reviews_df


def get_google_reviews(app_id, country, lang):
    reviews_dict = reviews_all(app_id,
                               sleep_milliseconds=0, # defaults to 0
                               country=country, # Changing this doesn't do anything - 3078 reviews show up regardless of what country code - this is probably all EN reviews, after checking how many per country with "app()"
                               lang=lang) # Just choosing EN filters out some english reviews, e.g. FR is mostly english)
    return reviews_dict

def google_reviews_to_df(reviews_dict, app_name):
        # Turn to dataframe
    col_date = []
    col_review = []
    col_title = []
    col_rating = []
    col_app = []
    for review in reviews_dict:
        col_date.append(review['at'])
        col_title.append('')
        col_review.append(review['content'])
        col_rating.append(review['score'])
        col_app.append(app_name)
    
    reviews_df = pd.DataFrame({'date':col_date,
                               'title':col_title,
                               'review':col_review,
                               'rating':col_rating,
                               'app':col_app})
    return reviews_df
    


def get_all_reviews()

    # Scrape apple reviews
    apple_reviews_economist = {}
    apple_reviews_espresso = {}
    for country in ['US','GB','CA','AU','ZA','IE']: # Arbitary list of english speaking countries
        apple_reviews_economist[country] = get_apple_reviews('the-economist', app_id='1239397626', country=country)
        apple_reviews_espresso[country] = get_apple_reviews('espresso-from-the-economist', app_id='896628003', country=country)

    # Scrape google reviews
    google_reviews_economist = {}
    google_reviews_espresso = {}
    google_reviews_economist['All'] = get_google_reviews('com.economist.lamarr', country='US', lang='EN') # I think this is is all english reviews worldwide
    google_reviews_espresso['All'] = get_google_reviews('com.economist.darwin', country='US', lang='EN') # I think this is is all english reviews worldwide


    # Turn to dataframe and combine
    all_reviews_list = []

    for r in apple_reviews_economist:
        all_reviews_list.append(apple_reviews_to_df(apple_reviews_economist[r], 'The Economist (Apple)'))
    for r in apple_reviews_espresso:
        all_reviews_list.append(apple_reviews_to_df(apple_reviews_espresso[r], 'Espresso (Apple)'))
    for r in google_reviews_economist:
        all_reviews_list.append(google_reviews_to_df(google_reviews_economist[r], 'The Economist (Google)'))
    for r in google_reviews_economist:
        all_reviews_list.append(google_reviews_to_df(google_reviews_espresso[r], 'Espresso (Google)'))

    final_reviews_df = pd.concat(all_reviews_list).reset_index(drop=True)

    return final_reviews_df

