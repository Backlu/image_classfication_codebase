# coding: utf-8

#History
# 4/13: first imp ver.

from utils import *
import os, sys, cv2, glob, datetime
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import operator
import joblib
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class Classifier(object):
    _defaults = {
        'model_ver': 'xx',
        'model_dir':None,
        'cls_map_path':None,
        'ensemble_models':[],
        'label_path':None,
        #'tta_n':3
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"        
        
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.__dict__.update(kwargs)

        
    def inference(self, df):
        model_cls_map = joblib.load(self.cls_map_path)        
        pred_list = []
        for model_name in self.ensemble_models:
            gen_df = self.get_data_gen(df, model_name)
            model_path = os.path.join(self.model_dir, f'{self.model_ver}/{model_name}_{self.model_ver}_full.h5')
            model = load_model(model_path, compile=False)
            #for i in range(self.tta_n):
            pred = model.predict(gen_df, verbose=1, workers=4, use_multiprocessing=True)
            #pred = model.predict(gen_df, verbose=1)
            pred_list.append(pred)
            pred_score = np.max(pred, axis=1)
            pred_label = np.argmax(pred, axis=1)
            pred_class = [model_cls_map[x] for x in pred_label]
            df[f'{model_name}_pred_score'] = pred_score
            df[f'{model_name}_pred_class'] = pred_class
            df[f'{model_name}_pred_class'] = df[f'{model_name}_pred_class'].astype(int)
                
        cols = df.columns
        cols = list(filter(lambda x: 'pred_class' in x, cols))
        pred_candidate = list(set([c.replace('_pred_class','') for c in cols]))    
        
        ret = df.apply(lambda r: self.voting(r, pred_candidate), axis=1)
        df['best_cls'], df['best_score'] = zip(*ret)    
        return df, pred_list

    def voting(self, r, pred_candidate):
        score_dict = {}
        cnt_dict = {}
        for t in pred_candidate:
            score = r[f'{t}_pred_score']
            cls = r[f'{t}_pred_class']
            score_dict[cls] = score_dict.get(cls,0)+score
            cnt_dict[cls] = cnt_dict.get(cls,0)+1
        best_cls = max(score_dict, key=score_dict.get)
        avg_score = score_dict[best_cls]/cnt_dict[best_cls]
        return best_cls, avg_score

    def get_data_gen(self, df, model_name):
        img_size = model_input_size[model_name][:2]
        predGen = ImageDataGenerator(
            samplewise_center=True, samplewise_std_normalization=True,
            vertical_flip=False, horizontal_flip=True,
            width_shift_range=0.1, height_shift_range=0.1,
            rotation_range=60, shear_range=0.15, 
            #channel_shift_range=0.2, 
            brightness_range=[0.9, 1.1],
            zoom_range =0.1,
            preprocessing_function=model_preprocess[model_name],
        )
        df['fake_class']='fake'
        gen_df = predGen.flow_from_dataframe(df, batch_size=25, directory=None, x_col='filepath', y_col='fake_class', shuffle=False, target_size=img_size)
        return gen_df
    
    def acc_rank(self, df):
        df_true = pd.read_csv(self.label_path)
        label_dict = dict(zip(df_true.filename, df_true.category))
        df['category'] = df['key'].map(lambda x: label_dict[x])
        cols = df.columns
        cols = list(filter(lambda x: 'pred_class' in x, cols))
        pred_candidate = list(set([c.replace('_pred_class','') for c in cols]))
        acc_dict = {}
        for t in pred_candidate:
            model_name = t.split('_')[0]
            acc = accuracy_score(df['category'], df[f'{t}_pred_class'])
            acclist = acc_dict.get(model_name,[])
            acclist.append(acc)
            acc_dict[model_name] = acclist
        model_acc = {k: np.mean(v) for k,v in acc_dict.items()}
        model_rank = sorted(model_acc.items(), key=operator.itemgetter(1), reverse=True)
        return model_rank
    
    def evaluate_acc(self, df):
        df_true = pd.read_csv(self.label_path)
        label_dict = dict(zip(df_true.filename, df_true.category))
        df['category'] = df['key'].map(lambda x: label_dict[x])
        ensemble_acc = accuracy_score(df['category'], df['best_cls'])
        print(f'Ensemble acc: {ensemble_acc:.4f}')

        
