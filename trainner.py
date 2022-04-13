# coding: utf-8

#History
# 4/13: first imp ver.

import os
import glob
import pandas as pd
import numpy as np
import datetime
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D, Activation
from tensorflow.keras.optimizers import SGD, Adagrad
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, CSVLogger, TensorBoard, ReduceLROnPlateau, LambdaCallback
from official.nlp import optimization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from cutmix_keras import CutMixImageDataGenerator
from utils import *

class Trainner(object):
    _defaults = {
        'tiral_ver': 'xx',
        'pretrain_dir':None,
        'cls_map_path':'model/model_cls_map.pkl',
        'model_dir':None,
        'cutmix':True,
        'cls_number':-1,
        'lr':3e-5,
        'multi_gpu':False
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
        
        
    def get_model(self, model_name):
        pretrain_weights = model_path[model_name]
        input_shape = model_input_size[model_name]
        MODEL = model_api[model_name]
        
        pretrain_weights = glob.glob(os.path.join(self.pretrain_dir, '*.h5'))        
        weights_apth = [x for x in pretrain_weights if model_name in x][0]
        if self.pretrain_dir==None:
            base_model=MODEL(weights=None, include_top=False, input_shape=input_shape) 
        else:
            try:
                base_model=MODEL(weights=weights_apth, include_top=False, input_shape=input_shape) 
            except:
                base_model=MODEL(weights=None, include_top=False, input_shape=input_shape) 
        
        x=base_model.output
        x=GlobalAveragePooling2D()(x)
        #x1=GlobalMaxPooling1D()(x)
        #x = Concatenate()([x, x1])
        x=Dense(1024, activation='relu')(x)
        x=Dense(256, activation='relu')(x)
        preds=Dense(self.cls_number)(x) 
        preds = Activation('softmax', name='prob')(preds)
        model=Model(inputs=base_model.input,outputs=preds)

        try:    
            model.load_weights(weights_apth)
        except:
            pass

        return model

    def get_data(self, df, model_name):
        input_shape = model_input_size[model_name]
        preprocessing = model_preprocess[model_name]
        imgAug_tr = ImageDataGenerator(
            samplewise_center=True, samplewise_std_normalization=True,
            vertical_flip=True, horizontal_flip=True,
            width_shift_range=0.15, height_shift_range=0.15,
            rotation_range=360, shear_range=0.2, 
            #channel_shift_range=0.2, 
            brightness_range=[0.8, 1.2],
            zoom_range =0.2,
            validation_split=0.2,
            preprocessing_function=preprocessing,
        )
        dgen_tr1 = imgAug_tr.flow_from_dataframe(df, batch_size=self.batch, directory=None, x_col='filepath', y_col='category_str', class_mode='categorical', shuffle=True, target_size=input_shape[:2], subset="training",)
        dgen_val = imgAug_tr.flow_from_dataframe(df, batch_size=self.batch, directory=None, x_col='filepath', y_col='category_str', class_mode='categorical', shuffle=True, target_size=input_shape[:2], subset="validation",)
        
        if self.cutmix:
            if 'isAug' in df.columns:
                df_aug = df[df['isAug']==True]
                sample_qty = len(df)-len(df_aug)
                if sample_qty>0:
                    df_sample = df_aug.sample(sample_qty, replace=True)
                    df_aug = pd.concat([df_aug, df_sample])
            else:
                df_aug = df
                
            dgen_tr2 = imgAug_tr.flow_from_dataframe(df_aug, batch_size=self.batch, directory=None, x_col='filepath', y_col='category_str', class_mode='categorical', shuffle=True, target_size=input_shape[:2], subset="training",)
            dgen_tr = CutMixImageDataGenerator(
                generator1=dgen_tr1, generator2=dgen_tr2,
                img_size=input_shape[0], batch_size=self.batch,
            )
        else:
            dgen_tr = dgen_tr1

        return dgen_tr, dgen_val

    def training(self, model_name, df, epoch=30):        
        batch = model_batch_size[model_name]
        self.batch = batch*2 if self.multi_gpu else batch
        tiral_ver = self.tiral_ver
        print('======',model_name,'========')
        model_dir = os.path.join(self.model_dir, tiral_ver)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        dgen_tr, dgen_val = self.get_data(df, model_name)
        model_cls_map = {v: k for k, v in gen_df.class_indices.items()}
        
        steps_per_epoch = dgen_tr.samples//self.batch
        steps_per_epoch_val = dgen_val.samples//self.batch
        num_train_steps = steps_per_epoch * epoch
        num_warmup_steps = int(0.1*num_train_steps)
        init_lr = self.lr
        opt = optimization.create_optimizer(init_lr=init_lr,
                                                  num_train_steps=num_train_steps,
                                                  num_warmup_steps=num_warmup_steps,
                                                  optimizer_type='adamw')

        model = self.get_model(model_name)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])
        for idx, layer in enumerate(model.layers):
            layer.trainable = True

        best_model_path = os.path.join(model_dir, f'{model_name}_{tiral_ver}_best.h5')
        final_model_path = os.path.join(model_dir, f'{model_name}_{tiral_ver}_full.h5')
        log_path = os.path.join(model_dir, f'{model_name}_{tiral_ver}.log')
        fig_path = os.path.join(model_dir, f'{tiral_ver}_{model_name}.jpg')
        log_dir = f"logs/scalars/{model_name}_{tiral_ver}_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        #tensorboard --logdir logs/scalars --bind_all

        
        checkpointer=ModelCheckpoint(monitor='val_loss',filepath=best_model_path, verbose=1, save_best_only=True, save_weights_only=True)
        early_stop = EarlyStopping(monitor='loss', min_delta=1e-6, patience=15, mode='min', verbose=1)
        csvLogger = CSVLogger(log_path)
        tboard = TensorBoard(log_dir=log_dir)
        callbacks = [checkpointer, csvLogger, early_stop, tboard]

        history = model.fit(dgen_tr, steps_per_epoch=steps_per_epoch, validation_steps=steps_per_epoch_val, epochs=epoch, validation_data=dgen_val, callbacks=callbacks, workers=10, use_multiprocessing=True) 

        model.load_weights(best_model_path)
        model.save(final_model_path)
        joblib.dump(model_cls_map, self.cls_map_path)
        
        plt.figure(figsize=(8,3))
        plt.subplot(121)
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        loss_min = np.min(history.history['loss'])
        plt.ylim(0,loss_min*12)
        plt.title(model_name)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.subplot(122)
        plt.plot(history.history['categorical_accuracy'], label='accuracy')
        plt.plot(history.history['val_categorical_accuracy'], label='val_accuracy')
        plt.title(model_name)
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.legend()
        plt.ylim((0, 1.2))
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.show()          
        
    def categorical_focal_loss(self, gamma=2., alpha=.25):
        def categorical_focal_loss_fixed(y_true, y_pred):
            y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
            epsilon = K.epsilon()
            y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
            cross_entropy = -y_true * K.log(y_pred)
            loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
            return K.sum(loss, axis=1)
        return categorical_focal_loss_fixed        