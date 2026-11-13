"""Generalized matrix-factorization adapter with implicit preference construction and popularity fallback."""
import os
import time
import datetime
import requests
import pandas as pd
import numpy as np
import math

from typing import Dict, Text, Tuple
from pymongo import MongoClient

import tensorflow as tf                 
#import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs  
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split

from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import collections


class ModelGMF():

      def __init__(self, config):
          # print('MODEL GMF -> INIT')
          self.config = config          
          self.data_duration = []       
          self.df_duration = None       
          
                                        
          self.pop_channels_rec = []    
          self.ind2name = {}            
          self.name2ind = {}            
          self.df_users = None          
          self.channel_corr_mat = None  

          self.log = []                 
          self.model = None             
          self.run_time = 0
          self.recommend_num = 50       

          self.files_path = self.config.path + '/ModelGMF'

          
          if not os.path.isdir(self.files_path):
              os.makedirs(self.files_path, mode=0o755, exist_ok=True)


      
      def fit(self, days=30):
          
          if os.path.isfile(self.files_path + '/status.log'): 
              os.remove(self.files_path + '/status.log')

          self.run_time = 0  

          
          answer = self.__get_data_duration(days)
          if answer['status'] != 'OK':
              return answer  

          
          answer = self.__data_processing_duration()
          if answer['status'] != 'OK':
              return answer  

          
          # self.__get_data_website(days)
          # if answer['status'] != 'OK':
          

          
          # answer = self.__data_processing_website()
          # if answer['status'] != 'OK':
          

          
          answer = self.__fit_model()
          if answer['status'] != 'OK':
              return answer  

          return answer



      
      def __get_data_duration(self,  days=30):
          # print('MODEL GMF -> GET DATA')

          start_time = time.time()
          from_time = int(time.time()) - int(days) * 86400
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
                      # print(i, len(self.data_duration), len(data_list))
                      if len(data_list) < self.config.api_page_size:  
                          # print(self.data_duration[0:5])
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

          
          
          # print(self.df_duration.head(10))
          return answer 


      
      def __get_data_website(self, days=30):
          # print('MODEL -> GET DATA')

          start_time = time.time()

          client = MongoClient(self.config.mongo_client_website)
          db = client['stats']
          website = db['website']

          
          timestamp = (int(time.time()) - int(days) * 86400) * 1000
          results = website.find({"timestamp": {"$gte":timestamp}})
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



      
      def __data_processing_duration(self):
          # print('MODEL -> DATA PROCESSING')
          start_time = time.time()

          df = self.df_duration
          df.dropna(inplace=True)

          
          self.popular_list = df.groupby(['channel_id']).size().sort_values(ascending=False)[0:100].tolist()

          
          
          view_mean = df.groupby(['channel_id']).size().mean()
          

          
          view_counter = pd.DataFrame({'Count' : df.groupby(['channel_id']).size()}).reset_index()
          k_1 = view_counter.shape[0]

          view_counter = view_counter.loc[view_counter['Count'] > view_mean/5]  # 1/5
          k_2 = view_counter.shape[0]

          

          reducer = df['channel_id'].isin(view_counter['channel_id'])
          df_2 = df[reducer]
          

          self.df_duration = df_2.drop_duplicates()
          

          delta_time = time.time() - start_time
          self.run_time += delta_time
          answer = {
              'status': 'OK', 
              'message': f'Data processed, number of rows: {self.df_duration.shape[0]}, execution time: {round(delta_time, 4)}s'
          }
          self.__logging(answer)

          return answer



      
      
      



      
      def __logging(self, answer):
          with open(self.files_path + '/status.log', 'a') as file:
              now = datetime.datetime.now()
              d = f'{now.year}-{now.month}-{now.day} {now.hour}:{now.minute}:{now.second}'
              text = f"{d}, {answer['status']}: {answer['message']} \n"
              file.write(text)



      
      def __fit_model(self):
          # print('MODEL -> FIT MODEL')
          start_time = time.time()

          'Prepare popularity fallback and personalized recommendation outputs.'

          
          
          
          result = self.df_duration.groupby(['channel_id','user_id']).sum().reset_index()
          result.drop(columns=['time'],axis = 1, inplace=True)
          
          
          
          result['rating'] = pd.DataFrame(np.log((result['duration'] - 1)/60000))

          
          def weighted_rating(v,m,R,C):
              '''
              Calculate the weighted rating
              
              Args:
              v -> average rating for each channel (float)
              m -> minimum votes required to be classified as popular (float)
              R -> average rating for the channel (pd.Series)
              C -> average rating for the whole dataset (pd.Series)
              
              Returns:
              pd.Series
              '''
              return ( (v / (v + m)) * R) + ( (m / (v + m)) * C )

          def assign_popular_based_score(rating_df, user_col, channel_col, rating_col):
              '''
              Assigned popular based score (based on the IMDB weighted average).
              
              Args:
              result -> pd.DataFrame contains ['channel_id', 'rating'] for each user.
              
              Returns
              popular_channels -> pd.DataFrame contains channel and weighted rating.
              '''
              
              # pre processing
              vote_count = (
                  rating_df
                  .groupby(channel_col,as_index=False)
                  .agg( {user_col:'count', rating_col:'mean'} )
                  )
              vote_count.columns = [channel_col, 'vote_count', 'avg_rating']
              
              # calcuate input parameters
              C = np.mean(vote_count['avg_rating'])
              m = np.percentile(vote_count['vote_count'], 80)               
              vote_count = vote_count[vote_count['vote_count'] >= m]
              R = vote_count['avg_rating']
              v = vote_count['vote_count']
              vote_count['weighted_rating'] = weighted_rating(v,m,R,C)
              
              # post processing
              popular_channels = vote_count.loc[:,[channel_col, 'vote_count', 'avg_rating', 'weighted_rating']]
              
              return popular_channels

          # init constant
          USER_COL = 'user_id'
          CHANNEL_COL = 'channel_id'
          RATING_COL = 'rating'

          # calcualte popularity based
          pop_channels = assign_popular_based_score(result, USER_COL, CHANNEL_COL, RATING_COL)
          pop_channels = pop_channels.sort_values('weighted_rating', ascending = False)

          
          self.pop_channels_rec = pop_channels['channel_id'].iloc[:self.recommend_num].astype(str).tolist()
          


          
          
          
          result['channel_indx'] = pd.factorize(result['channel_id'])[0]
          result['user_indx'] = pd.factorize(result['user_id'])[0]

          
          result.sort_values('channel_indx', ascending=True, inplace=True)
          df_channels = pd.DataFrame(result.sort_values('channel_indx', ascending=True)['channel_id'].unique())
          df_channels.columns = ['channel_id']
          df_channels['channel_id'] = df_channels['channel_id'].astype(str)

          result.sort_values('user_indx', ascending=True, inplace=True)
          self.df_users = pd.DataFrame(result.sort_values('user_indx', ascending=True)['user_id'].unique())
          self.df_users.columns = ['user_id']
          self.df_users['user_id'] = self.df_users['user_id'].astype(str)   

          
          self.ind2name = df_channels.to_dict()['channel_id']                      
          self.name2ind = {v:k for k,v in self.ind2name.items()}                   

          
          result['user_indx'] = result['user_indx'].astype(str)
          result['channel_indx'] = result['channel_indx'].astype(str)


          # ----------------
          
          def df_to_ds(df):

              
              ds = tf.data.Dataset.from_tensor_slices(
                  (dict(df[['user_indx','channel_indx']]), df['rating']))
              
              
              ds = ds.map(lambda x, y: {
                  'user_indx' : x['user_indx'],
                  'channel_indx' : x['channel_indx'],
                  'rating' : y
                  })

              return ds.batch(256)

          class RankingModel(keras.Model):

              def __init__(self, users_db, user_indx, channel_indx, embedding_size):
                  super().__init__()
                  
                  # user model
                  input = keras.Input(shape=(), dtype=tf.string)
                  x = keras.layers.StringLookup(
                      vocabulary = users_db.index.astype(str).to_numpy(), mask_token = None
                      )(input)
                  output = keras.layers.Embedding(
                      input_dim = len(user_indx) + 1,
                      output_dim = embedding_size,
                      name = 'embedding'
                      )(x)
                  self.user_model = keras.Model(inputs = input,
                                                outputs = output,
                                                name = 'user_model')
                  
                  # channel model
                  input = keras.Input(shape=(), dtype=tf.string)
                  x = keras.layers.StringLookup(
                      vocabulary = df_channels.index.astype(str).to_numpy(), mask_token = None
                      )(input)
                  output = keras.layers.Embedding(
                      input_dim = len(channel_indx) + 1,
                      output_dim = embedding_size,
                      name = 'embedding'
                      )(x)
                  self.channel_model = keras.Model(inputs = input,
                                            outputs = output,
                                            name = 'channel_model')

                  # rating model
                  user_input = keras.Input(shape=(embedding_size,), name='user_emb')
                  channel_input = keras.Input(shape=(embedding_size,), name='channel_emb')
                  x = keras.layers.Concatenate(axis=1)([user_input, channel_input])
                  x = keras.layers.Dense(256, activation = 'relu')(x)
                  x = keras.layers.Dense(64, activation = 'relu')(x)
                  output = keras.layers.Dense(1)(x)
                  
                  self.rating_model = keras.Model(
                      inputs = {
                          'user_indx' : user_input,
                          'channel_indx' : channel_input
                          },
                      outputs = output,
                      name = 'rating_model'
                      )

              def call(self, inputs: Dict[Text, tf.Tensor]) -> tf.Tensor:

                  user_emb = self.user_model(inputs['user_indx'])
                  channel_emb = self.channel_model(inputs['channel_indx'])

                  prediction = self.rating_model({
                      'user_indx' : user_emb,
                      'channel_indx' : channel_emb
                      })
                  
                  return prediction

          class GMFModel(tfrs.models.Model):

              def __init__(self, users_db, user_indx, channel_indx, embedding_size):
                  super().__init__()
                  self.ranking_model = RankingModel(users_db, user_indx, channel_indx, embedding_size)
                  self.task = tfrs.tasks.Ranking(
                      loss = keras.losses.MeanSquaredError(),
                      metrics = [keras.metrics.RootMeanSquaredError()]
                      )
              
              def call(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:
                  
                  return self.ranking_model(
                      {
                      'user_indx' : features['user_indx'], 
                      'channel_indx' : features['channel_indx']
                      })

              def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:

                  return self.task(labels = features.pop('rating'),
                                  predictions = self.ranking_model(features))

          
          train, test = train_test_split(result[['user_indx','channel_indx','rating']], train_size = .8, random_state=42)
          train, val_ds = train_test_split(train, train_size = .8, random_state=None)

          
          train, test, val_ds = df_to_ds(train), df_to_ds(test),  df_to_ds(val_ds)
          
          # ----------------
          
          embedding_size = 64
          self.model = GMFModel(self.df_users, self.df_users.index.astype(str),
                          df_channels.index.astype(str),
                          embedding_size)
          self.model.compile(
              optimizer = keras.optimizers.Adagrad(learning_rate = .1)
              )

          
          history = self.model.fit(train, epochs=30, verbose=1, validation_data=val_ds, validation_steps=10)
          
          eval_result = self.model.evaluate(test, return_dict=True, verbose=0)
          
          # print(eval_result)

          
          channel_emb = self.model.ranking_model.channel_model.layers[-1].get_weights()[0]
          
          self.channel_corr_mat = cosine_similarity(channel_emb)

          # ----------------
          delta_time = time.time() - start_time
          self.run_time += delta_time
          message = 'Model training completed, '
          message += f'total execution time: {round(self.run_time)}s '
          answer = {
              'status': 'OK',
              'message': message
          }
          self.__logging(answer)

          return answer



      
      def predict(self, user_id):
          

          def top_k_channels(channel_id, top_k, corr_mat, map_name):
              
              
              top_channels = corr_mat[channel_id,:].argsort()[-top_k-1:][::-1] 
              top_channels = [map_name[e] for e in top_channels][1:]
              return top_channels

          
          if user_id in self.df_users['user_id'].unique():
              
              
              df = self.df_duration[self.df_duration['user_id']==int(user_id)][['time', 'channel_id']].sort_values('time', ascending=False)

              
              ch_id = str(df['channel_id'].iloc[0])

              
              
              try:
                  answer = top_k_channels(self.name2ind[ch_id],
                                          top_k = self.recommend_num,
                                          corr_mat = self.channel_corr_mat,
                                          map_name = self.ind2name)
              except:
                  
                  answer = self.pop_channels_rec

          else:
              
              answer = self.pop_channels_rec
              

          
          return answer
