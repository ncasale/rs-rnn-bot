import urllib
import requests
from pandas.io.json import json_normalize
import json
import pandas as pd
import datetime
import time
import math
import numpy as np
import random

from neural_network import NeuralNetwork

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import Activation, Dense
from keras.models import Sequential

if __name__ == '__main__':
    #Get a list of all items
    df = pd.read_json(path_or_buf='https://rsbuddy.com/exchange/summary.json',orient='index', convert_axes=True)
    df = df[['id','name','buy_average','buy_quantity','sell_average','sell_quantity','overall_average','overall_quantity']]
    data = df.sort_values(by=['id']).reset_index()
    data = data.drop(labels='index',axis=1)

    #Run this to populate a csv: itemKey
    itemKey = data[['id', 'name']]
    itemKey.to_csv(path_or_buf='./itemKey.csv', columns=('id','name'), index=False)

    #Create test set of specified range number of items
    data = pd.read_csv('./itemKey.csv', skiprows=[])
    startNum = 0
    numItems = 3
    inputdata = data[startNum:startNum+numItems]
    inputdata

    #Use this to pull a specific items by name
    names = ['Armadyl godsword','Fire rune', 'Soft clay']

    #read the itemKey csv and parse the key for associated item ids
    data = pd.read_csv('./itemKey.csv', skiprows=[])
    inputdata = pd.DataFrame()
    for name in names:
        inputdata = inputdata.append(data.loc[data['name'] == name])
    inputdata = inputdata.reset_index().drop(labels='index',axis=1)

    print("Query List:")
    print(inputdata)

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
    for index, row in urlquery.iterrows():
        row['data'] = row['data'][['ts','buyingPrice','buyingCompleted','sellingPrice','sellingCompleted','overallPrice','overallCompleted']]
        row['data'] = row['data'].sort_values(by=['ts'],ascending=1).reset_index()
        count = 0 
        for ind, r in row['data'].iterrows():
            r['ts'] = r['ts']/1000
            row['data'].loc[ind, 'ts'] = int(r['ts'])
            row['data'].loc[ind, 'date'] = datetime.datetime.fromtimestamp(r['ts']).isoformat()
        row['data'] = row['data'].drop(labels='index',axis=1)

    test_set_size = 30 #In Days - define the number of epochs to include in the test data

    #create test_item object to log training and test sets
    #test_item = pd.DataFrame(columns=['id', 'name', 'train_x', 'train_y', 'test_x', 'test_y', 'pred_y'])
    test_item = pd.DataFrame(columns=['id', 'name', 'train', 'test', 'pred_y'])
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
        train = test_set.iloc[0]['data'].iloc[0:math.ceil(test_set_size*6)].values[:,0:6]
        # train = test_set.iloc[0]['data'].iloc[0:math.ceil(test_set_size*6)].values[:,5]
        print ("The shape of the " + test_set.iloc[0]['name'] +  " input array is: "+ str(train.shape))

        #create test set to estimate next 6 prices (next day of prices)
        test = test_set.iloc[0]['data'].iloc[math.ceil(test_set_size*6):math.ceil(test_set_size*6)+6].values[:,0:6]

        #test key to compare to Y-hat
        #test_y = test_set.iloc[0]['data'].iloc[math.ceil(test_set_size*6):math.ceil(test_set_size*6)+6].values[:,5]
        #test_item = test_item.append({'id':item_id, 'name': test_set.iloc[0]['name'], 'train_x':train_x, 'train_y':train_y, 'test_x':test_x, 'test_y': test_y}, ignore_index=True)
        test_item = test_item.append({'id':item_id, 'name': test_set.iloc[0]['name'], 'train': train, 'test':test}, ignore_index=True)
        print("Record completed at test_item position [" + str(h) + "]")
        h += 1
        #TODO: Get this all in a for loop to iterate over multiple entries

    #Test creation of Neural Net
    nn = NeuralNetwork()
    nn.generateInitialNetwork([1,1])
    nn.calculateNodeActivations()
    print(nn.outputNode.getActivation())
    print(test_item.iloc[0]['train'])




    #everything below this point is for testing the network size/structure. This uses the Keras library within TensorFlow.


    
    def create_model(layers,activation):
        model = Sequential()
        for i, nodes in enumerate(layers):
            if i==0:
                model.add(Dense(nodes,input_dim=train_x.shape[1]))
                model.add(Activation(activation))
            else: 
                model.add(Dense(nodes))
                model.add(Activation(activation))
        model.add(Dense(1)) #Note: no activations present beyond this point

        model.compile(optimizer='adadelta', loss='mse')
        return model
    model = KerasRegressor(build_fn=create_model, verbose = 0)
    layers = [[16], [4,2], [4], [16,4]]
    activations = ['tanh', 'relu']
    param_grid = dict(layers=layers, activation=activations, batch_size = [42, 180], epochs=[6])
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error')
