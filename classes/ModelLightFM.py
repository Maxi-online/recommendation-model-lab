"""LightFM adapter for implicit collaborative recommendation."""
import os
import pandas as pd
import numpy  as np
import time
import datetime
import requests
from pymongo import MongoClient
from lightfm import LightFM
from scipy.sparse import coo_matrix


class ModelLightFM():

      def __init__(self, config):
          # print('MODEL LightFM -> INIT')
          self.config = config  
          self.data_duration = []  
          self.df_duration = None  
          self.df_website = None  
          self.log = []  
          self.model = None  
          self.run_time = 0
          self.files_path = self.config.path
          self.channel_key = [] 
          self.channel_value = [] 
          self.channel_dict = {} 
          self.channel_dict_rev = {} 
          self.user_key = [] 
          self.user_value = [] 
          self.user_dict = {} 
          self.user_dict_rev = {} 
          self.coo_data = None 
          self.popular_channels = [] 

      
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

      
      def __get_data_duration(self, days=30):
          # print('MODEL LightFM -> GET DATA')

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
          results = website.find({"timestamp": {"$gte": timestamp}})
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

      
      def __logging(self, answer):
          with open(self.files_path + '/status.log', 'a') as file:
              now = datetime.datetime.now()
              d = f'{now.year}-{now.month}-{now.day} {now.hour}:{now.minute}:{now.second}'
              text = f"{d}, {answer['status']}: {answer['message']} \n"
              file.write(text)

      
      def __data_processing_website(self):
          # print('MODEL -> DATA PROCESSING')
          start_time = time.time()

          website = self.df_website

          website['streamId'] = website['streamer'].fillna(website['streamId'])
          website.drop(columns=['user','streamer'], inplace=True)
          website.dropna(subset=['event'], inplace=True)
          website['game'] = website['game'].fillna('unknoun_game')
          website['message'] = website['message'].fillna('unknoun_message')
          website['streamStarted'] = website['streamStarted'].fillna(0)
          website[['streamId','streamStarted','userId']] = website[['streamId','streamStarted','userId']].astype('int64')
          website = website.loc[website.userId != 0]
          website.drop(columns=['message'], inplace=True)
          website = website.loc[website.event != 'chat-message']
          website = website.loc[website.event != 'stream-follow']
          website = website.loc[website.streamStarted != 0]
          view = website.loc[website.event == 'stream-view']
          view.drop(columns=['event'], inplace=True)
          view = view.reindex(columns=['game', 'streamId', 'userId', 'streamStarted', 'timestamp'])
          # dt = pd.to_datetime(view.timestamp, unit='ms').min()
          # view = view.loc[view.streamStarted > int(datetime.timestamp(datetime(dt.year, dt.month, dt.day, 0, 0, 0)))]
          view_agg = view.groupby(['userId',
                                   'streamId',
                                   'game',
                                   'streamStarted'], as_index=False).agg({'timestamp':['min','max']})
          view_agg.columns = ['_'.join(col).rstrip('_') for col in view_agg.columns.values]
          view_agg['duration'] = view_agg.timestamp_max - view_agg.timestamp_min
          view_agg.drop(columns=['timestamp_max'], inplace=True)
          view_agg = view_agg.rename(columns={'timestamp_min': 'timestamp'})
          view_agg = view_agg.loc[view_agg.duration >= 60_000]
          view_agg.duration = view_agg.duration // 1000
          view_agg.index = np.arange(len(view_agg))

          self.user_key = pd.unique(view_agg.userId)
          self.user_value = [i for i in range(len(self.user_key))]
          self.user_dict = dict(zip(self.user_key, self.user_value))
          self.user_dict_rev = dict(zip(self.user_value, self.user_key))
          view_agg['us_id'] = view_agg['userId'].map(self.user_dict)

          self.channel_key = pd.unique(view_agg.streamId)
          self.channel_value = [i for i in range(len(self.channel_key))]
          self.channel_dict = dict(zip(self.channel_key, self.channel_value))
          self.channel_dict_rev = dict(zip(self.channel_value, self.channel_key))
          view_agg['ch_id'] = view_agg['streamId'].map(self.channel_dict)

          self.coo_data = coo_matrix((view_agg.duration, (view_agg.us_id, view_agg.ch_id)))

          self.popular_channels = list(view_agg.sort_values(by = 'duration', ascending = False))

          delta_time = time.time() - start_time
          self.run_time += delta_time
          answer = {
              'status': 'OK',
              'message': f'Data processed, number of rows: {self.df_duration.shape[0]}, execution time: {round(delta_time, 4)}s'
          }
          self.__logging(answer)

          

          return answer

      
      def __data_processing_duration(self):
          # print('MODEL -> DATA PROCESSING')
          start_time = time.time()

          
          self.popular_list = self.df_duration.groupby(['channel_id']).size().sort_values(ascending=False)[0:100].tolist()

          durations = self.df_duration

          
          durations.dropna(axis = 0, inplace=True)

          
          user_id_max = 1e+7 # 100.000.000
          durations.drop(durations[durations.user_id > user_id_max].index, axis=0, inplace=True)

          
          duration_max = int(np.timedelta64(8, 'h') / np.timedelta64(1, 'ms'))
          durations.loc[durations['duration'] > duration_max, 'duration'] = duration_max

          
          duration_min = int(pd.Timedelta(1, "m") / np.timedelta64(1, 'ms'))
          durations.drop(durations[durations.duration < duration_min].index, axis=0, inplace=True)

          
          self.channel_key   = pd.unique(durations.channel_id) 
          self.channel_value = [i for i in range(len(self.channel_key))] 
          self.channel_dict  = dict(zip(self.channel_key, self.channel_value)) 
          self.channel_dict_rev = dict(zip(self.channel_value, self.channel_key)) 
          durations['ch_id'] = durations['channel_id'].map(self.channel_dict) 

          
          self.user_key   = pd.unique(durations.user_id) 
          self.user_value = [i for i in range(len(self.user_key))] 
          self.user_dict  = dict(zip(self.user_key, self.user_value)) 
          self.user_dict_rev = dict(zip(self.user_value, self.user_key)) 
          durations['us_id'] = durations['user_id'].map(self.user_dict) 

          
          durations = durations.groupby(['user_id',
                                         'channel_id']).agg({'time':'max',
                                                             'duration':'sum',
                                                             'ch_id':'max',
                                                             'us_id':'max'})

          # sorted_channels = durations.sort_values(by = 'duration', ascending = False)
          # self.popular_channels = [self.channel_dict_rev[id] for id in sorted_channels.ch_id]
          groupby_channels = durations.groupby(['ch_id']).agg({'duration':'sum', 'ch_id':'max'})
          sorted_channels = groupby_channels.sort_values(by = 'duration', ascending = False)
          self.popular_channels = [self.channel_dict_rev[id] for id in sorted_channels.ch_id]

          self.coo_data = coo_matrix((durations.duration, (durations.us_id, durations.ch_id)))

          delta_time = time.time() - start_time
          self.run_time += delta_time
          answer = {
              'status': 'OK',
              'message': f'Data processed, number of rows: {self.df_duration.shape[0]}, execution time: {round(delta_time, 4)}s'
          }
          self.__logging(answer)

          

          return answer

      def __fit_model(self):
          # print('MODEL -> FIT MODEL')
          start_time = time.time()

          self.model = LightFM(no_components=32, loss='warp')
          self.model.fit(self.coo_data, epochs=4, verbose=True)

          
          # with open(modelpath + 'model.pickle', 'wb') as fle:
          #     pickle.dump(model, fle)

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

      
      # def __load_model(self, modelpath):
      #
      #     with open(modelpath + 'model.pickle', 'rb') as fle:
      #         model = pickle.load(fle)
      #
      
      #
      #     return model

      def predict(self, user_id):
          n = 10
          user_id = int(user_id)
          try:
              col = self.channel_value
              us_id = self.user_dict[user_id]
              row = np.array([us_id for _ in range(len(col))])
              pred = self.model.predict(row, col, num_threads=os.cpu_count())
              list_pred = list(zip(col, pred))
              sort_pred = sorted(list_pred, reverse=True, key=lambda x: x[1])
              rec_ch_id = [first for first, second in sort_pred][:n]
              rec_channel_ids = [self.channel_dict_rev[id] for id in rec_ch_id]
          except KeyError:
              rec_channel_ids = self.popular_channels[:n]
          except ValueError:
              rec_channel_ids = self.popular_channels[:n]

          return rec_channel_ids
