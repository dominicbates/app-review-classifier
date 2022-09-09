import pandas as pd
import numpy as np
from app_store_scraper import AppStore
from google_play_scraper import reviews_all, reviews, Sort, app
import time



def get_apple_reviews(app_name, app_id, country, max_retries=10):
    
    # Try and get scrape reviews...
    for n in range(max_retries+1):
        reviews = AppStore(country = country, app_name = app_name, app_id = app_id)
        #reviews.review(how_many=100000)
        reviews.review()
        reviews_dict = reviews.reviews
        del reviews
        
        # If no results, retry after a few seconds
        if len(reviews_dict) < 2:
            print("No results found for",country,"- let's wait a sec, then retry...")
            time.sleep(10+(n*2)) # Increase wait each time up to 30s
        # If multiple of 20 results, show warning
        elif (len(reviews_dict)%20)==0:
            print('Found '+str(len(reviews_dict))+' results for '+country+'!..')
            print('WARNING: Multiple of 20 so possible HTTP request blocked! Moving on anyway though... :/')
            break
        # If not multiple of 20 results, probably all good!
        else:
            print('Found '+str(len(reviews_dict))+' results for '+country+'! Moving on! :)') # Sometimes, this doesn't work either and outputs a multiple of 20
            break

    if len(reviews_dict) < 2:
        print('ERROR: Tried',max_retries+1,'times and didn\'t get any response... :(')

    return reviews_dict



def apple_reviews_to_df(reviews_dict, app_name):
    
    # Turn to dataframe
    col_date = []
    col_review = []
    col_title = []
    col_rating = []
    col_app = []
    col_unique_id = []
    col_os = []
    for review in reviews_dict:
        col_date.append(review['date'])
        col_title.append(review['title'])
        col_review.append(review['review'])
        col_rating.append(review['rating'])
        col_unique_id.append('apple_'+app_name+'_'+review['userName'])
        col_app.append(app_name)
        col_os.append('iOS')
    
    reviews_df = pd.DataFrame({'date':col_date,
                               'title':col_title,
                               'review':col_review,
                               'rating':col_rating,
                               'app':col_app,
                               'os':col_os,
                               'uniqueid':col_unique_id})
    return reviews_df



def get_google_reviews(app_id, country, lang):

    # Try multiple times if request fails
    for n in range(10):
        try:
            reviews_dict = reviews_all(app_id,
                                       sleep_milliseconds=0, # defaults to 0
                                       country=country, # Changing this doesn't do anything - 3078 reviews show up regardless of what country code - this is probably all EN reviews, after checking how many per country with "app()"
                                       lang=lang) # Just choosing EN filters out some english reviews, e.g. FR is mostly english)
            break
        except:
            time.sleep(10+(n*2)) 


    return reviews_dict



def google_reviews_to_df(reviews_dict, app_name):
    # Turn to dataframe
    col_date = []
    col_review = []
    col_title = []
    col_rating = []
    col_app = []
    col_unique_id = []
    col_os = []
    for review in reviews_dict:
        col_date.append(review['at'])
        col_title.append('')
        col_review.append(review['content'])
        col_rating.append(review['score'])
        col_unique_id.append('google_'+app_name+'_'+review['userName'])
        col_app.append(app_name)
        col_os.append('Android')
    
    reviews_df = pd.DataFrame({'date':col_date,
                               'title':col_title,
                               'review':col_review,
                               'rating':col_rating,
                               'app':col_app,
                               'os':col_os,
                               'uniqueid':col_unique_id})
    return reviews_df
    


def get_all_reviews(country_list = None, max_retries = 10):

    # Load default countries
    if country_list is None:
        countries_english = ['US','GB','CA','AU','ZA','IE','NZ'] # English speaking countries
        countries_other = ['IN','ID','BR','DE','ES','NL','SE','SG'] # Populous countries where vast majority of reviews are in english
        countries_list = countries_english + countries_other

    # Scrape apple reviews
    apple_reviews_economist = {}
    apple_reviews_espresso = {}
    count = 0

    print('Getting apple reviews...')
    for country in countries_list: 

        apple_reviews_economist[country] = get_apple_reviews('the-economist', app_id='1239397626', country=country, max_retries=max_retries)
        if count <= 2: # Sleep between calls for the big ones
            time.sleep(30) 
        else:
            time.sleep(10) 

        apple_reviews_espresso[country] = get_apple_reviews('espresso-from-the-economist', app_id='896628003', country=country, max_retries=max_retries)
        if count <= 2: # Sleep between calls for the big ones
            time.sleep(30) 
        else:
            time.sleep(10) 
        count+=1
    print('Apple reviews downloaded!')

    # Scrape google reviews
    google_reviews_economist = {}
    google_reviews_espresso = {}
    print('Getting google reviews...')
    google_reviews_economist['All'] = get_google_reviews('com.economist.lamarr', country='US', lang='EN') # I think this is is all english reviews worldwide
    google_reviews_espresso['All'] = get_google_reviews('com.economist.darwin', country='US', lang='EN') # I think this is is all english reviews worldwide
    print('Google reviews downloaded!')

    # Turn to dataframe and combine
    all_reviews_list = []

    for r in apple_reviews_economist:
        all_reviews_list.append(apple_reviews_to_df(apple_reviews_economist[r], 'The Economist'))
    for r in apple_reviews_espresso:
        all_reviews_list.append(apple_reviews_to_df(apple_reviews_espresso[r], 'Espresso'))
    for r in google_reviews_economist:
        all_reviews_list.append(google_reviews_to_df(google_reviews_economist[r], 'The Economist'))
    for r in google_reviews_economist:
        all_reviews_list.append(google_reviews_to_df(google_reviews_espresso[r], 'Espresso'))

    final_reviews_df = pd.concat(all_reviews_list).reset_index(drop=True)

    return final_reviews_df



def update_reviews_df(old_reviews_df, new_reviews_df=None):
    '''
    Gets all reviews, and then adds any new reviews to existing dataframe
    This function exists as sometimes countries can fail mid-way, meaning it outputs
    only a subsample of total reviews. This ensures, reviews will be picked up next time 
    the function is run, and if a country fails, but the review is already in the old dataframe, 
    it will still be kept
    '''
    if new_reviews_df is None:
        print('Getting new reviews to update dataframe...')
        new_reviews_df = get_all_reviews(country_list = None)
        new_reviews_df['row_created_date'] = pd.Timestamp.now()
    else:
        print('Updating reviews using supplied dataframe (check that "row_created_date" column is defined)...')
    
    # Combine dataframes
    print('Combining with old reviews dataframe...')
    combined_df = pd.concat([old_reviews_df, new_reviews_df]).reset_index(drop=True)

    # Remove any duplictaed rows (i.e. share the same text and datetime) - can't use username since doesn't exist for some people
    combined_df = combined_df.sort_values(['review', 'date', 'row_created_date'],
                                       ascending = [True, False, True]).reset_index(drop=True)
    combined_df = combined_df.groupby(['review','date'],as_index=False).first().reset_index(drop=True)
    
    
    print('\nOriginal reviews dataframe length:',len(old_reviews_df))
    print('Extracted reviews dataframe length:',len(new_reviews_df))
    print('Overlapping reviews:',len(combined_df))
    return combined_df
