import urllib
import requests
from pandas.io.json import json_normalize
import json
import pandas as pd
import datetime
import time
import math
import numpy as np

from neural_network import NeuralNetwork

#libraries for testing network sizes/activation functions
"""
FUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import Activation, Dense
from keras.models import Sequential
"""


def getItems(): #Get a list of all items
    df = pd.read_json(path_or_buf='https://rsbuddy.com/exchange/summary.json',orient='index', convert_axes=True)
    df = df[['id','name','buy_average','buy_quantity','sell_average','sell_quantity','overall_average','overall_quantity']]
    data = df.sort_values(by=['id']).reset_index()
    data = data.drop(labels='index',axis=1)
    #Run this to populate a csv: itemKey
    itemKey = data[['id', 'name']]
    itemKey.to_csv(path_or_buf='./itemKey.csv', columns=('id','name'), index=False)


def generateInputByItemNumber(startNum,numItems): 
    """Pull items by range of Index values"""
    allItems = pd.read_csv('./itemKey.csv', skiprows=[])
    inputdata = allItems[startNum:startNum+numItems]
    print("Query List:")
    print(inputdata)
    return inputdata

def generateInputByItemName(names):  #Pull a specific items by name
    """Pull specific items by name"""
    data = pd.read_csv('./itemKey.csv', skiprows=[])
    inputdata = pd.DataFrame()
    for name in names:
        inputdata = inputdata.append(data.loc[data['name'] == name])
    inputdata = inputdata.reset_index().drop(labels='index',axis=1)
    print("Query List:")
    print(inputdata)
    return inputdata

def getItemRecordsFromURL(inputdata, test_set_size):
    #Iterate through all items, grab data from api and append to urlquery dataframe
    i = 0
    firstIteration = True
    urlquery = pd.DataFrame(columns=['id', 'name', 'data'])
    while(i<inputdata['id'].count()):
        key = inputdata.iloc[i]['id']
        name = inputdata.iloc[i]['name']
        if (firstIteration == True):
            print("Item Id - " + str(key) + ": " + name)
        #Attempt to get data from API, retry if HTTP error
        try:
            url = f'https://api.rsbuddy.com/grandExchange?a=graph&g=240&start=1474615279000&i={key}'
            tempDataFrame = pd.DataFrame()
            tempDataFrame = pd.read_json(path_or_buf=url,orient='records', convert_axes=False)
            urlquery = urlquery.append({'id':key, 'name': name, 'data':tempDataFrame}, ignore_index=True)
            i+=1
            #check if there are more items to run. If no more items, return message indicating the process is done
            if (i == inputdata['id'].count()):
                print("It worked! That was the last item!")
            else:
                print("It worked! On to the next one.")
            firstIteration = True
        except:
            print("Got fucked. Trying again. :)")
            time.sleep(1)
            firstIteration = False
    #Reformat urlquery columns, convert timestamp from miliseconds to seconds, convert timestamp to date, and sort by date
    print("Processing item records...")
    for index, row in urlquery.iterrows():
        row['data'] = row['data'][['ts','buyingPrice','buyingCompleted','sellingPrice','sellingCompleted','overallPrice','overallCompleted']]
        row['data'] = row['data'].sort_values(by=['ts'],ascending=1).reset_index()
        count = 0 
        for ind, r in row['data'].iterrows():
            r['ts'] = r['ts']/1000
            row['data'].loc[ind, 'ts'] = int(r['ts'])
            row['data'].loc[ind, 'date'] = datetime.datetime.fromtimestamp(r['ts']).isoformat()
        row['data'] = row['data'].drop(labels='index',axis=1)
    print("Done!")
    return urlquery

def generateTrainAndTestPercentages(urlquery):
    #Iterate through each row of urlquery and generate percentage for buy price, buy completed, sell price, overall price
    for index, row in urlquery.iterrows():
        for colName in ['buyingPricePer', 'buyingCompletedPer', 'sellingPricePer', 'sellingCompletedPer', 'overallPricePer', 'overallCompletedPer']:
            row['data'][colName] = pd.Series(np.random.randn(len(row['data']['ts'])), index=row['data'].index)
        #Each row represents and item, which has an associated dataframe 
        for ind, itemDataRow in row['data'].iterrows():
            #Use data from last row to calc percentage change
            for colName in ['buyingPrice', 'buyingCompleted', 'sellingPrice', 'sellingCompleted', 'overallPrice', 'overallCompleted']:
                if(ind==0):
                    row['data'].loc[ind, colName+'Per'] = 0
                else:
                    row['data'].loc[ind, colName+'Per'] = calculatePercentageChange(row['data'], colName, ind)
                
                
            
def calculatePercentageChange(dataframe, columnName, index):
    """Calculate delta from last time stamp"""
    return (dataframe.loc[index, columnName] - dataframe.loc[index-1, columnName])/dataframe.loc[index-1, columnName]


def createTrainAndTestSet(urlquery):
    print("Creating Test and Training Datasets...")
    #create test_item object to log training and test sets
    test_item = pd.DataFrame(columns=['id', 'name', 'train_val', 'test_val', 'train_per', 'test_per', 'pred_per'])
    h = 0
    #iterate through the urlquery results to generate training sets
    while(h<urlquery['id'].count()):  
        item_id = urlquery.iloc[h]['id'] #Define the item to be queried for
        print("Item ID: " + str(item_id))
        #takes the item_id and generates the test data for the specified parameters as an array
        test_set = urlquery.loc[urlquery['id'] == item_id]
        train_x_headers = list(urlquery.iloc[0]['data'].columns.values[0:6])
        
        #for column key, uncomment this line below:
        #print(train_x_headers)

        #create training datasets
        train_val = test_set.iloc[0]['data'].iloc[0:math.ceil(test_set_size*6)].values[:,0:6]
        print ("The shape of the " + test_set.iloc[0]['name'] +  " input array is: "+ str(train_val.shape))

        #create test set to estimate next 6 prices (next day of prices)
        test_val = test_set.iloc[0]['data'].iloc[math.ceil(test_set_size*6):math.ceil(test_set_size*6)+6].values[:,0:6]

        #Generate deltas
        generateTrainAndTestPercentages(urlquery)

        #DEBUG -- output to CSV
        for index, row in urlquery.iterrows():
            row['data'].to_csv(path_or_buf='./urlquery' + row['name'] + '.csv', columns=('ts','buyingPrice', 'buyingPricePer', 'buyingCompleted', 'buyingCompletedPer', 'sellingPrice',  'sellingPricePer', 'sellingCompleted', 'sellingCompletedPer', 'overallPrice', 'overallPricePer', 'overallCompleted', 'overallCompletedPer'), index=True)

        #test key to compare to Y-hat
        test_item = 0
        #test_item = test_item.append({'id':item_id, 'name': test_set.iloc[0]['name'], 'train_val': train_val, 'test_val': test_val, 'train_per': train_per, 'test_per': test_per}, ignore_index=True)
        print("Record completed at test_item position [" + str(h) + "]")
        h += 1
        #TODO: Get this all in a for loop to iterate over multiple entries
        return test_item

if __name__ == '__main__':
    # Get a list of all items
    # getItems()

    #Pull items by range of Index values
    inputdata = generateInputByItemNumber(0,3)
    
    #Pull specific items by name
    # inputdata = generateInputByItemName(['Armadyl godsword','Fire rune', 'Soft clay'])

    #In Days - define the number of epochs to include in the test data
    test_set_size = 30    

    urlquery = getItemRecordsFromURL(inputdata, test_set_size)

    #Create test_item object to log training and test sets
    test_item = createTrainAndTestSet(urlquery)

    #Test creation of Neural Net
    nn = NeuralNetwork()
    nn.generateInitialNetwork([1,1])
    nn.calculateNodeActivations()
    print(nn.outputNode.getActivation())

    #TODO: Create Year time stamp for seasonality
    #TODO: Create time from today time stamp
    #TODO: Create training and test sets
    #TODO: Test model size


    #random shit
    #import keras.backend as K
    """
    train_val = test_item.iloc[2]['train_val'] # test data. not real
    train_y = test_item.iloc[2]['train_y'] # test data. not real

    
    def create_model(layers,activation):
        model = Sequential()
        for i, nodes in enumerate(layers):
            if i==0:
                model.add(Dense(nodes,input_dim=train_val.shape[1]))
                model.add(Activation(activation))
            else: 
                model.add(Dense(nodes))
                model.add(Activation(activation))
        model.add(Dense(1)) #Note: no activations present beyond this point

        model.compile(optimizer='adadelta', loss='mse')
        return model

    print('creating model')
    model = KerasRegressor(build_fn=create_model, verbose = 0)
    layers = [[16], [4,2], [4], [16,4]]
    activations = ['tanh', 'relu']
    param_grid = dict(layers=layers, activation=activations, batch_size = [42, 180], epochs=[6])
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error')

    # grid_result = grid.fit(train_x, train_y) testing for network size """