"""Neural collaborative filtering adapter with user/item features and historical preprocessing. The portable benchmark is separate from this legacy data-source integration."""
from pymongo import MongoClient
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import datetime
import time
import requests 


from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#tf/keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import matplotlib.pyplot as plt
import os


class ModelNCF():
  def __init__(self, config):
        # print('MODEL NFC -> INIT')
        #super(ModelNCF, self).__init__(**kwargs)
        self.config = config  
        self.data_duration = []  
        self.df_duration = False  
        self.df_website = False  
        self.log = []  

        self.k = 10
        self.embedding_size = 50 
        self.batch_size = 64 
        self.epochs = 100 
        self.seed = 42
        self.path = '/content/drive/My Drive/internship/production'

        self.df_duration = None  
        self.df_website = None  
        self.df = None
        self.log = []  
        self.model = None  

        self.run_time = 0
        self.files_path = self.config.path + '/ModelNCF'


        
        if not os.path.isdir(self.files_path):
            os.makedirs(self.files_path, mode=0o755, exist_ok=True)



    
  def fit(self, days=30):
        
        if os.path.isfile(self.files_path + '/status.log'): 
            os.remove(self.files_path + '/status.log')

        self.run_time = 0  

        
        answer = self.__get_data_duration(days)
        if answer['status'] != 'OK':
            return answer  

        
        # self.__get_data_website(days) 

        
        answer = self.__data_processing()
        if answer['status'] != 'OK':
            return answer  

        
        answer = self.__fit_model(self.embedding_size)
        if answer['status'] != 'OK':
            return answer  

        return answer

    
  def __get_data_duration(self,  days=30):
        start_time = time.time()
        from_time = int(time.time()) - int(days)*86400
        from_time_url = f'&from={from_time}'
        to_time = int(time.time())
        from_to_time = f'&to={to_time}'

        
        for i in range(1, 1000):
            url = self.config.interactions_api_url + from_time_url + from_to_time + '&page=' + str(i)
            try:
                api_req = requests.get(url, timeout=30)
                if int(api_req.status_code) != 200:
                    answer = {
                        'status': 'ERROR', 
                        'message': f'Error connecting to the interaction-source API. Server API response code: {api_req.status_code}'
                    }
                    self.__logging(answer)
                    break
                else:
                    
                    data_list = api_req.json()
                    self.data_duration.extend(data_list)
                    if len(data_list) < self.config.api_page_size:  
                        self.df_duration = pd.DataFrame.from_dict(self.data_duration)
                        break
            except:
                answer = {
                    'status': 'ERROR', 
                    'messare': 'Error connecting to the interaction-source API'
                }
                self.__logging(answer)
                break

        
        delta_time = time.time() - start_time
        if (len(self.data_duration) > 0):
            answer = {
                'status': 'OK', 
                'message': f'Data received, number of rows: {len(self.data_duration)}, execution time: {round(delta_time, 4)}s'
            }
            self.__logging(answer)
        else:
            answer = {
                'status': 'ERROR', 
                'message': f'No data, execution time: {round(delta_time, 4)}s'
            }
            self.__logging(answer)
        self.data_duration = []  

        self.run_time += delta_time

        return answer 


    
  def __get_data_website(self, days=30):
        # print('MODEL -> GET DATA')

        start_time = time.time()

        client = MongoClient(self.config.mongo_client_website)
        db = client['stats']
        website = db['website']

        
        timestamp = (int(time.time()) - int(days) * 86400)*1000
        results = website.find({"timestamp":{"$gte":timestamp}})
        res = [r for r in results]
        self.df_website = pd.DataFrame(list(res))
        # print(self.df_website.head())

        
        delta_time = time.time() - start_time
        

        answer = {
            'status': 'OK', 
            'message': f'Data received, number of rows: {self.df_website.shape[0]}, execution time: {round(delta_time, 4)}s'
        }
        self.__logging(answer)
        return answer

    
  def __data_processing(self):
        'Prepare the adapter feature table from loaded interactions.'
        
        start_time= time.time()

        df = self.df_duration
        df['channel_id'] = pd.to_numeric(df['channel_id'], errors='coerce') 
        #df['user_id'] = pd.to_numeric(df['user_id'], errors='coerce')
        df = df.dropna()

        
        self.popular_list = df.groupby(['channel_id']).size().sort_values(ascending=False)[0:100].tolist()

        

        x = np.nan_to_num(df)
        Clust_dataset=StandardScaler().fit_transform(x)
        clasterNum = 11 
        k_means = KMeans(init="k-means++", n_clusters=clasterNum, n_init=12)
        k_means.fit(Clust_dataset)
        labels=k_means.labels_
        df_labels = pd.DataFrame(labels, columns=['cluster'])
        indexes = df.index
        df_labels.index = indexes
        df_clust = pd.merge(left=df, right=df_labels, left_index=True, right_index=True)
        df_clust = df_clust.drop(columns=['duration'])

        
        
        df_clust = df_clust.reindex(columns=['user_id', 'channel_id', 'cluster', 'time',]) # time  channel_id  user_id  duration
        df_clust.rename(columns={'user_id':'userId', 'channel_id':'itemId', 'cluster':'rating', 'time':'timestamp'}, inplace = True)
        df_clust['userId'] = df_clust['userId'].astype(int)
        df_clust['rating'] = df_clust['rating'].astype(int)
        df_clust['timestamp'] = df_clust['timestamp'].astype(int)
        df_clust = df_clust.reset_index()
        df = df_clust.copy()
        user_ids = df['userId'].unique().tolist() 
        self.user2user_encoded = {x: i for i, x in enumerate(user_ids)} 
        userencoded2user = {i: x for i, x in enumerate(user_ids)} 
        streamer_ids = df['itemId'].unique().tolist() 
        self.streamer2streamer_encoded = {x: i for i, x in enumerate(streamer_ids)} 
        self.streamer_encoded2streamer = {i: x for i, x in enumerate(streamer_ids)} 

        
        df['user'] = df['userId'].map(self.user2user_encoded) 
        df['streamer'] = df['itemId'].map(self.streamer2streamer_encoded)

        self.num_users = len(self.user2user_encoded) 
        self.num_streamers = len(self.streamer_encoded2streamer) 
        df['rating'] = df['rating'].values.astype(np.float32) 
        df['userId'] = df['userId'].values.astype(np.float32) 
        df['itemId'] = df['itemId'].values.astype(np.float32) 
        self.df_duration = df
        delta_time = time.time() - start_time
        self.run_time += delta_time 

        answer = {
            'status': 'OK',
            'message': f'Data processed, number of rows: {df.shape[0]}, execution time: {round(delta_time, 4)}s', #self.df.shape[0]
            'data': self.df_duration,
            'num_users': self.num_users,
            'num_streamers': self.num_streamers,
            'streamer2streamer_encoded': self.streamer2streamer_encoded,
            'user2user_encoded': self.user2user_encoded,
            'streamer_encoded2streamer': self.streamer_encoded2streamer,
        }
        
        self.__logging(answer)

        return answer
        
  def __fit_model(self, embedding_size):
        start_time = time.time()
        x_train, x_val, y_train, y_val = self.__train_split()
        self.model = CreateModel(self.num_users, self.num_streamers, embedding_size=self.embedding_size)
        self.model.compile(
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                        )
        history = self.model.fit(x_train,
                        y_train,
                        batch_size=self.batch_size,
                        epochs=self.epochs,
                        verbose=1,
                        validation_data=[x_val, y_val],
                        )
        
        message =   'Model training completed, '

        delta_time = time.time() - start_time
        self.run_time += delta_time
              

        answer = {
                  'status': 'OK', 
                  'message': message,
              }

        self.__logging(answer)
              
        return answer

  def model_save(model, path):
        model.save(path)
        return 

  def model_load(path):
        return keras.models.load_model(path)

    
  def __logging(self, answer):
        with open(self.files_path + '/status.log', 'a') as file:
            now = datetime.datetime.now()
            d = f'{now.year}-{now.month}-{now.day} {now.hour}:{now.minute}:{now.second}'
            text = f"{d}, {answer['status']}: {answer['message']} "
            file.write(text)



    
  def predict(self, user_id):
        data = self.df_duration
        model = self.model

        
        
        l_temp = []
        for i in data.userId:
          l_temp.append(i)

        k = 10 
        if user_id in l_temp: 
          streamers_watched_by_user = data[data.userId == user_id] 
          
          streamers_not_watched = data[~data["itemId"].isin(streamers_watched_by_user.itemId.values)]["itemId"]
          streamers_not_watched = list(set(streamers_not_watched).intersection(set(self.streamer2streamer_encoded.keys())))
          streamers_not_watched = [[self.streamer2streamer_encoded.get(x)] for x in streamers_not_watched]

          user_encoder = self.user2user_encoded.get(user_id) 
          user_streamer_array = np.hstack(([[user_encoder]] * len(streamers_not_watched), streamers_not_watched)) 
          ratings = model.predict(user_streamer_array).flatten() 
          top_ratings_indices = ratings.argsort()[-k:][::-1] 

          
          recommended_streamer_ids = [self.streamer_encoded2streamer.get(streamers_not_watched[x][0]) for x in top_ratings_indices] 
          
          answer = []
          for i in recommended_streamer_ids:
            answer.append(str(i))

        else: 
          
          answer = [] 
          #temp = data.groupby(data.columns.streamerId.to_list(), as_index=False).size()[:10]
          temp = data.groupby(data.itemId.to_list(), as_index=False).size()[:k]

          for i in range(len(temp)):
            answer.append(temp['index'].iloc[i])
        return answer       
       
  def __train_split(self):
        df = self.df_duration
        df = df.sample(frac=1, random_state=42) 
        x = df[['user', 'streamer']].values 
        df['rating'] = df['rating'].values.astype(np.float32) 
        
        min_rating = min(df['rating'])
        max_rating = max(df['rating'])
        
        y = df['rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values 
        
        train_indices = int(0.9 * df.shape[0])
        x_train, x_val, y_train, y_val = (
            x[:train_indices],
            x[train_indices:],
            y[:train_indices],
            y[train_indices:],
        )


        return x_train, x_val, y_train, y_val  



class CreateModel(keras.Model):
  def __init__(self, num_users, num_streamers, embedding_size, **kwargs): #, embedding_size, batch_size, epochs, **kwargs
        super(CreateModel, self).__init__(**kwargs)
        self.num_streamers = num_streamers
        self.embedding_size = embedding_size
        self.user_embedding = layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.user_bias = layers.Embedding(num_users, 1)
        self.streamer_embedding = layers.Embedding(
            num_streamers,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.streamer_bias = layers.Embedding(num_streamers, 1)
  
  def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        streamer_vector = self.streamer_embedding(inputs[:, 1])
        streamer_bias = self.streamer_bias(inputs[:, 1])
        dot_user_streamer = tf.tensordot(user_vector, streamer_vector, 2)
        
        x = dot_user_streamer + user_bias + streamer_bias
        
        return tf.nn.sigmoid(x)
