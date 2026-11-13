"""fastai collaborative-filtering adapter with artifact persistence and precomputed recommendations."""
import os
import sys
import time
import datetime
from datetime import datetime
import requests
import pandas as pd
import numpy as np
from typing import Dict, Text
from pymongo import MongoClient

import torch, fastai
from fastai.collab import *
from fastai.tabular.all import *

from recommenders.models.fastai.fastai_utils import cartesian_product, score

# print("System version: {}".format(sys.version))
# print("Pandas version: {}".format(pd.__version__))
# print("Fast AI version: {}".format(fastai.__version__))
# print("Torch version: {}".format(torch.__version__))
# print("Cuda Available: {}".format(torch.cuda.is_available()))
# print("CuDNN Enabled: {}".format(torch.backends.cudnn.enabled))


class ModelFastai:
    def __init__(self, config):
        # print('MODEL Fastai -> INIT')
        self.config = config  
        self.data_duration = []  
        self.df_duration = None  
        self.df_final = None 
        
        self.log = []  
        self.model = None  
        self.run_time = 0
        self.files_path = self.config.path + '/ModelFastai'

        
        if not os.path.isdir(self.files_path):
            os.mkdir(self.files_path, mode=0o755)


    
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

        
        answer = self.fit_model()
        if answer['status'] != 'OK':
            return answer  

        
        # answer = self.compute_predicts()
        # if answer['status'] != 'OK':
        

        return answer


    
    def predict(self, user_id):
        start_time = time.time()
        # self.model = load_learner(self.files_path+'/model.pkl')
        n = 10 
        

        _, total_items = self.model.classes.values()
        total_items = total_items[1:]
        total_users = [str(user_id)]

        users_items = cartesian_product(np.array(total_users),np.array(total_items))
        users_items = pd.DataFrame(users_items, columns=['userID','itemID'])

        dl_test = self.model.dls.test_dl(users_items, with_labels=True)
        preds = self.model.get_preds(dl=dl_test, with_decoded=True)
        users_items['preds'] = preds[2]
        # users_items.sort_values(by='preds', ascending=False).head(10)

        delta_time = time.time() - start_time

        # print(f'predict execution time: {round(delta_time, 4)}s')

        channel_list=users_items[users_items["userID"]==str(user_id)].sort_values(by='preds', ascending=False).head(n)

        return channel_list['itemID'].tolist()


    
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

        self.df_final = df_2.drop_duplicates()
        

        
        self.df_final=self.df_final.groupby(['user_id', 'channel_id']).sum().reset_index()
        self.df_final['userID']=self.df_final['user_id'].astype(str)
        self.df_final['itemID']=self.df_final['channel_id'].astype(str)
        self.df_final=self.df_final.drop(columns=['user_id'],axis = 1)
        self.df_final=self.df_final.drop(columns=['channel_id'],axis = 1)

        

        self.df_final['rating'] = np.where(self.df_final['duration'] < 10*1000*60, 1.0, 
                                          
                                  np.where(self.df_final['duration'] < 30*1000*60, 2.0, 
                                          
                                  np.where(self.df_final['duration'] < 60*1000*60, 3.0,
                                          
                                  np.where(self.df_final['duration'] < 120*1000*60,4.0,5.0))))

        
        
        
        
        

        
        self.df_final=self.df_final.drop(columns=['duration'],axis = 1)
        self.df_final=self.df_final.drop(columns=['time'],axis = 1)

        delta_time = time.time() - start_time
        self.run_time += delta_time

        answer = {
            'status': 'OK', 
            'message': f'Data processed, number of rows: {self.df_final.shape[0]}, execution time: {round(delta_time, 4)}s'
        }
        self.__logging(answer)
        return answer


    
    def __logging(self, answer):
        with open(self.files_path + '/status.log', 'a') as file:
            now = datetime.now()
            d = f'{now.year}-{now.month}-{now.day} {now.hour}:{now.minute}:{now.second}'
            text = f"{d}, {answer['status']}: {answer['message']} \n"
            file.write(text)



    
    def fit_model(self):
        # print('MODEL Fastai -> FIT MODEL')
        start_time = time.time()

        dls = CollabDataLoaders.from_df(self.df_final)

        # embs = get_emb_sz(dls)

        # model = DotProductBias(n_users, n_movies, 50)
        # learn = Learner(dls, model, loss_func=MSELossFlat())
        # learn.fit_one_cycle(5, 5e-3, wd=0.1)

        
        self.model = collab_learner(dls, n_factors=64, y_range=[0,5.5], wd=1e-1)
        

        
        self.model.fit_one_cycle(15, 5e-3, wd=0.1)

        
        self.model.export(self.files_path+'/model.pkl')

        delta_time = time.time() - start_time
        self.run_time += delta_time

        answer = {
            'status': 'OK', 
            'message': f"Took {round(self.run_time)} seconds for training."
        }
        self.__logging(answer)

        return answer

    # def __cartesian_product(*arrays):
    #     """Compute the Cartesian product in fastai algo. This is a helper function.

    #     Args:
    #         arrays (tuple of numpy.ndarray): Input arrays

    #     Returns:
    #         numpy.ndarray: product

    #     """
    #     la = len(arrays)
    #     dtype = np.result_type(*arrays)
    #     arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    #     for i, a in enumerate(np.ix_(*arrays)):
    #         arr[..., i] = a
    #     return arr.reshape(-1, la)

    
    
    # def compute_predicts(self):
    #     # print('MODEL Fastai -> COMPUTE PREDICTS')
    #     start_time = time.time()

    #     self.model = load_learner(self.files_path, 'model.pkl')

    

    #     # Get all users and items that the model knows

    #     total_users, total_items = self.model.data.train_ds.x.classes.values()
    #     total_items = total_items[1:]
    #     total_users = total_users[1:]

    #     users_items = cartesian_product(np.array(total_users),np.array(total_items))
    #     users_items = pd.DataFrame(users_items, columns=[USER,ITEM])

    #     with Timer() as test_time:
    #         top_k_scores = score(self.model, 
    #                             test_df=users_items,
    #                             user_col=USER, 
    #                             item_col=ITEM, 
    #                             prediction_col=PREDICTION,
    #                             top_k=n)

    #     # # print("Took {} seconds to compute {} predictions.".format(test_time, len(users_items)))
    #     # channel_list=top_k_scores[top_k_scores["userID"]==str(user_id)].sort_values(by='prediction', ascending=False)

    #     # # print(channel_list['itemID'].tolist())
    #     top_k_scores.to_csv(self.files_path+'/users_items_predicts.csv')

    #     delta_time = time.time() - start_time
    #     self.run_time += delta_time

    #     answer = {
    #         'status': 'OK', 
    #         'message': f"Took {round(self.run_time)} seconds for predicting."
    #     }
    #     self.__logging(answer)

    #     return answer
