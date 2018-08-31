import urllib
import requests
from pandas.io.json import json_normalize
import json
import pandas as pd
pd.options.mode.chained_assignment = None
import datetime
import time
import pytz
import math
import numpy as np

from neural_network import NeuralNetwork

"""
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import Activation, Dense
from keras.models import Sequential
"""


def generate_item_records_from_summary():
    """ 
        Generate item records using the summary.json link from the OSBuddy API
    """
    df = pd.read_json(path_or_buf='https://rsbuddy.com/exchange/summary.json',orient='index', convert_axes=True)
    df = df[['id','name','buy_average','buy_quantity','sell_average','sell_quantity','overall_average','overall_quantity']]
    data = df.sort_values(by=['id']).reset_index()
    data = data.drop(labels='index',axis=1)
    
    #Output item id/name pairs to a csv file
    item_key = data[['id', 'name']]
    file_name = './item_key.csv'
    item_key.to_csv(path_or_buf=file_name, columns=('id','name'), index=False)


def generate_input_data_by_item_number(start_num, num_items): 
    """
        Pull items by range of Index values
        
            @param start_num: the starting index 
            @param num_items: the number of items to generate data for, starting from
                              the start_num
    """
    file_name = './item_key.csv'
    all_items = pd.read_csv(file_name, skiprows=[])
    items = all_items[start_num: start_num + num_items]
    return items

def generate_input_data_by_item_name(names): 
    """
        Pull specific items by name
        
            @param names: a list containing the item names to generate data for
    """
    file_name = './item_key.csv'
    data = pd.read_csv(file_name, skiprows=[])
    items = pd.DataFrame()
    for name in names:
        items = items.append(data.loc[data['name'] == name])
    items = items.reset_index().drop(labels='index',axis=1)
    return items

def get_item_records_from_url(input_data, frequency):
    """ 
        Iterate through all items, grab data from api and append to dataframe 
        
            @param input_data: a list of items with their associated index
            @param frequency: the period at which data is sampled, in minutes
    """
    i = 0
    item_lookup_failed = False
    item_records = pd.DataFrame(columns=['id', 'name', 'data'])
    while(i < input_data['id'].count()):
        key = input_data.iloc[i]['id']
        name = input_data.iloc[i]['name']
        #Print item information only on first data retrieval attempt 
        if not item_lookup_failed:
            print('Querying API for ' + name + ' data. ' + 'Item Id: ' + str(key))
        try:
            #Attempt to get data from API, retry if HTTP error
            url = f'https://api.rsbuddy.com/grandExchange?a=graph&g={frequency}&start=1474615279000&i={key}'
            temp_df = pd.DataFrame()
            temp_df = pd.read_json(path_or_buf=url, orient='records', convert_axes=False)
            item_records = item_records.append({'id':key, 'name': name, 'data':temp_df}, ignore_index=True)
            item_lookup_failed = False
            i+=1
        except:
            print("Retrying...")
            time.sleep(1) #Avoid getting blacklisted by API
            item_lookup_failed = True
            
    print('Item retrieval complete!')

    #Add correct formatting to item record dates/times
    item_records = format_item_record_dates(item_records)

    return item_records

def format_item_record_dates(item_records):
    """ 
        Converts timestamp to Unix seconds, tacks on formatted date field, and creates Unix 
        seconds from most recent datapoint, and Unix seconds position from Jan-1.
        
            @param item_records: the master item data structure
    """
    #Get most recent timestamp in all the data (highest value) 
    most_recent_ts = 0
    for row in item_records['data']:
        max_ts = row['ts'].max()
        if max_ts > most_recent_ts:
            most_recent_ts = max_ts
    
    #Get the timestamp of Jan-1 for 2015, 2016, 2017, and 2018
    jan1_timestamps = {}
    time_adjust = 18000 #Seconds ahead of EST
    for year in [2016, 2017, 2018]:
        jan1 = datetime.date(year, 1, 1)
        jan1_ts = time.mktime(jan1.timetuple())
        jan1_timestamps[str(year)] = jan1_ts - time_adjust

    for row in item_records['data']:
        #Convert timestamp from milliseconds to seconds
        row['ts'] = row['ts'] / 1000
        #Append column to indicate time delta from most recent record in dataset
        row['tsFromCurrent'] = most_recent_ts - row['ts']
        #Append column that converts ts to datetime object
        row['date'] = pd.to_datetime(row['ts'], unit='s')
        row['year'] = row['date'].dt.strftime('%Y')
        #Append a column representing the YTD seconds
        row['tsYtd'] = row['ts'] - pd.Series([jan1_timestamps[year] for year in row['year']])
        row.drop(columns=['year'], inplace=True)

            
    return item_records

def generate_train_and_test_deltas(item_records):    
    """
        Calculate percent difference from previous record for each relevant column. 
            
            @param item_records: the master item data structure
    """
    #Iterate through each item, and calculate percentages for their data
    for _, row in item_records.iterrows():
        #Create temp dataframe for percent changes
        pct_change_df = row['data'][['buyingPrice', 'buyingCompleted', 'sellingPrice', 
                            'sellingCompleted', 'overallPrice', 'overallCompleted']].copy()
        pct_change_df = pct_change_df.rename(index=int, 
                                         columns={'buyingPrice':'buyingPricePer', 'buyingCompleted':'buyingCompletedPer',
                                        'sellingPrice':'sellingPricePer', 'sellingCompleted':'sellingCompletedPer',
                                        'overallPrice':'overallPricePer', 'overallCompleted':'overallCompletedPer'})
        #Calculate percent change
        pct_change_df = pct_change_df.pct_change()
        
        #Join percent change df back to original dataframe
        row['data'] = row['data'].join(pct_change_df)


def createTrainAndTestSet(item_records, start_stamp, epoch_size, periods_per_epoch_unit):
    """
        Create the training and testing data sets for a particular epoch starting at start_stamp, and extending
        epoch_size units into the future.
        
            @param item_records: master item data structure
            @param start_stamp: the starting time stamp of this epoch
            @param epoch_size: the epoch size in days
            @param periods_per_epoch_unit: the number of periods ahead of the last training record
                that we will use this training data to predict
    """
    print("Creating Test and Training Datasets...")
    # Create dataframe to hold training/test data for the epoch
    epoch_dataframe = pd.DataFrame(columns=['id', 'name', 'train_data', 'test_data', 'pred_data'])
    record_index = 0
    end_stamp = start_stamp + math.ceil(epoch_size * periods_per_epoch_unit) 
    
    # Iterate through the each item in item_records
    while(record_index < item_records['id'].count()): 
        #Get item id
        item_id = item_records.iloc[record_index]['id'] 
        
        # Find all item records where ID matchest item_id
        test_set = item_records.loc[item_records['id'] == item_id]
        
        # The time stamp of the value we'll use this training data to predict
        predict_stamp = end_stamp + periods_per_epoch_unit
        
        #Ensure that the value we are attempting to predict exists for testing purposes
        if predict_stamp < len(test_set.iloc[0]['data']):
            # Create training dataset
            train_df = test_set.iloc[0]['data'].iloc[start_stamp:end_stamp]
            train_data = train_df[train_df.columns[-9:]][1:]          
            
            # Get test value -- the price 6 periods from the end of the training set (24 hours) 
            price_col = 14
            test_data = test_set.iloc[0]['data'].iloc[predict_stamp][price_col]

            # Append this item's data to the overall dataset for this epoch
            epoch_dataframe = epoch_dataframe.append({'id':item_id, 'name': test_set.iloc[0]['name'], 'train_data': train_data, 'test_data': test_data}, ignore_index=True)
            
        record_index += 1

    return epoch_dataframe

if __name__ == '__main__':
    #---------------------------------#
    #------ GENERATE INPUT DATA ------#
    #---------------------------------#
    #Get a list of all items
    generate_item_records_from_summary()
    #Pull specific items by name
    input_data = generate_input_data_by_item_name(['Leather','Dragon boots', 'Rune arrow'])

    #---------------------------------#
    #----- GENERATE ITEM RECORDS -----#
    #---------------------------------#
    item_records = get_item_records_from_url(input_data, 240)

    #----------------------------------#
    #----- CONVERT DATA TO DELTAS -----#
    #----------------------------------#
    generate_train_and_test_deltas(item_records)

    #-----------------------------------------------#
    #------- CREATE TRAINING AND TEST DATASETS -----#
    #-----------------------------------------------#
    # Define the number of epochs to include in the test data
    periods_per_epoch_unit = 6 #Each value is a 4 hour record
    start_stamp = 0
    epoch_size = 30
    num_epochs = 3

    #Create test_item object to log training and test sets
    test_train_set = pd.DataFrame(columns=['epoch_id', 'item_data'])

    #Calculate number of epochs required

    #Get item data for each epoch
    for i in range(num_epochs):
        curr_epoch = pd.DataFrame(columns=['epoch_id', 'item_data'])
        curr_epoch = curr_epoch.append([{'epoch_id': i, 'item_data': createTrainAndTestSet(item_records, start_stamp, epoch_size, periods_per_epoch_unit)}])
        test_train_set = test_train_set.append(curr_epoch)




