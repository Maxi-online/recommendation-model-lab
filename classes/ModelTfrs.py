"""TensorFlow Recommenders two-tower retrieval adapter. Optional TensorFlow/TFRS runtime; independently validate splits and metrics before application use."""
import os
import time
import datetime
import requests
import pandas as pd
import numpy as np
from typing import Dict, Text
# from pymongo import MongoClient

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs


class ModelTfrs():
    def __init__(self, config):
        self.config = config  
        self.data_duration = []  
        self.df_duration = None  
        self.df_website = None  
        self.log = []  
        self.model = None  
        self.run_time = 0
        self.files_path = self.config.path + '/ModelTfrs'
        self.popular_list = []  

        
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

        
        answer = self.__fit_model()
        if answer['status'] != 'OK':
            return answer  

        return answer


    
    def predict(self, user_id):
        
        index = tfrs.layers.factorized_top_k.BruteForce(self.model.user_model)
        
        index.index_from_dataset(
            tf.data.Dataset.zip((self.chanells_tensor.batch(10000), self.chanells_tensor.batch(10000).map(self.model.movie_model)))
        )

        
        _, titles = index(tf.constant([user_id]))
        answer = titles.numpy().astype('str').tolist()

        return answer[0]


    
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
                    # print(i, len(self.data_duration), len(data_list))
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



    
    def __data_processing(self):
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
        start_time = time.time()

        
        users_id_arr = self.df_duration['user_id'].unique().astype('bytes')

        
        chanells_id_arr = self.df_duration['channel_id'].unique().astype('bytes')
        self.chanells_tensor = tf.data.Dataset.from_tensor_slices(chanells_id_arr)

        
        ratings_ser = self.df_duration['user_id']
        df_ratings = pd.DataFrame(ratings_ser)
        df_ratings['channel_id'] = self.df_duration['channel_id']
        ratings_b_arr = df_ratings.to_numpy().astype('bytes')
        ratings = tf.data.Dataset.from_tensor_slices({"channel_id": ratings_b_arr[:, 1], "user_id": ratings_b_arr[:, 0]})

        
        tf.random.set_seed(42)
        shuffled = ratings.shuffle(df_ratings.shape[0], seed=42, reshuffle_each_iteration=False)

        
        if len(shuffled) > 6000 :
            train_num = 5000
            test_num = 1000
        else:
            train_num = round(len(shuffled) * 0.8)
            test_num = len(shuffled) - train_num

        
        train = shuffled.take(train_num)
        test = shuffled.skip(train_num).take(test_num)

        
        embedding_dimension = 64  

        
        user_model = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=users_id_arr, mask_token=None),
            tf.keras.layers.Embedding(len(users_id_arr) + 1, embedding_dimension)
        ])

        
        movie_model = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=chanells_id_arr, mask_token=None),
            tf.keras.layers.Embedding(len(chanells_id_arr) + 1, embedding_dimension)
        ])

        
        
        metrics = tfrs.metrics.FactorizedTopK(
            
            candidates = self.chanells_tensor.batch(10000).map(movie_model)
        )

        
        task = tfrs.tasks.Retrieval(
            metrics=metrics
        )

        self.model = CreateModel(user_model, movie_model, task)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))

        cached_train = train.shuffle(100_000).batch(32).cache()
        cached_test = test.batch(16).cache()

        
        history = self.model.fit(cached_train, epochs=18, verbose=0)

        delta_time = time.time() - start_time
        self.run_time += delta_time
        message =   'Model training completed, '
        message +=  f'top_1: {history.history["factorized_top_k/top_1_categorical_accuracy"][-1]:.4f}, '
        message +=  f'top_5: {history.history["factorized_top_k/top_5_categorical_accuracy"][-1]:.4f}, '
        message +=  f'top_10: {history.history["factorized_top_k/top_10_categorical_accuracy"][-1]:.4f}, '
        message +=  f'top_50: {history.history["factorized_top_k/top_50_categorical_accuracy"][-1]:.4f}, '
        message +=  f'top_100: {history.history["factorized_top_k/top_100_categorical_accuracy"][-1]:.4f}, '
        message +=  f'total execution time: {round(self.run_time)}s '
        answer = {
            'status': 'OK', 
            'message': message
        }
        self.__logging(answer)

        return answer






class CreateModel(tfrs.Model):
    def __init__(self, user_model, movie_model, task):
        super().__init__()
        self.movie_model: tf.keras.Model = movie_model
        self.user_model: tf.keras.Model = user_model
        self.task: tf.keras.layers.Layer = task


    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        
        user_embeddings = self.user_model(features["user_id"])
        
        
        positive_movie_embeddings = self.movie_model(features["channel_id"])

        
        return self.task(user_embeddings, positive_movie_embeddings)
