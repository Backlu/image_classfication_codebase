# coding: utf-8
from tensorflow.keras.applications import MobileNet, MobileNetV2, DenseNet121, DenseNet169, DenseNet201, NASNetLarge, Xception, ResNet101, ResNet101V2, InceptionResNetV2, ResNet152V2, VGG16
#from keras.applications.efficientnet import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3
from tensorflow.keras.applications.densenet import preprocess_input as den_preprocess_input
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess_input
from tensorflow.keras.applications.xception import preprocess_input as xce_preprocess_input
from tensorflow.keras.applications.resnet import preprocess_input as res_preprocess_input
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resv2_preprocess_input
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as incres_preprocess_input
from tensorflow.keras.applications.nasnet import preprocess_input as nas_preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mb_preprocess_input
from tensorflow.keras.applications.mobilenet import preprocess_input as mb_preprocess_input

ensemble_models = ['densenet201', 'densenet169',  'xception','inception_resnet_v2','resnet152v2','resnet101v2']

model_path = {'resnet152v2': 'resnet152v2_weights_tf_dim_ordering_tf_kernels_notop.h5',
              'densenet169':'densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5',
              'densenet201':'densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5',
              'densenet121':'densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5',
              'resnet101':'resnet101_weights_tf_dim_ordering_tf_kernels_notop.h5',
              'xception':'xception_weights_tf_dim_ordering_tf_kernels_notop.h5',
              'resnet101v2' : 'resnet101v2_weights_tf_dim_ordering_tf_kernels_notop.h5',
              'inception_resnet_v2' : 'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5',
              'NASNetLarge':'NASNet-large-no-top.h5',
              'mobilenetv2':'mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5',
              'mobilenetv3L':'weights_mobilenet_v3_large_224_1.0_float_no_top.h5',
              'mobilenetv3S':'weights_mobilenet_v3_small_224_1.0_float_no_top.h5',
              'efficientnetb0':'efficientnetb0_notop.h5',
              'efficientnetb1':'efficientnetb1_notop.h5',
              'efficientnetb2':'efficientnetb2_notop.h5',
              'efficientnetb3':'efficientnetb3_notop.h5',
              'inceptionv3':'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
              'vgg16':'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
             }
model_input_size = {'densenet169':(256, 256, 3),
              'densenet201':(256, 256, 3),
              'densenet121':(256, 256, 3),
              'resnet101':(256, 256, 3),     
              'xception':(256, 256, 3),
              'resnet101v2':(256, 256, 3),                    
              'inception_resnet_v2':(256, 256, 3),
              'resnet152v2':(256, 256, 3),
              'NASNetLarge':(331, 331, 3),
              'vgg16':(256, 256, 3),      
             }
model_api = {'densenet169':DenseNet169,
              'densenet201':DenseNet201,
              'densenet121':DenseNet121,     
             'resnet101':ResNet101,
              'xception':Xception,
              'resnet101v2':ResNet101V2,                    
              'inception_resnet_v2':InceptionResNetV2,
              'resnet152v2':ResNet152V2,
             'NASNetLarge':NASNetLarge,
             'mobilenetv2':MobileNetV2,
             'vgg16': VGG16,
             }
model_preprocess ={'densenet169':den_preprocess_input,
              'densenet201':den_preprocess_input,
              'densenet121':den_preprocess_input,
             'resnet101':res_preprocess_input,
              'xception':xce_preprocess_input,
              'resnet101v2':resv2_preprocess_input,
              'inception_resnet_v2':incres_preprocess_input,
              'resnet152v2':resv2_preprocess_input,
              'NASNetLarge':nas_preprocess_input,
              'mobilenetv2':mb_preprocess_input,
              'vgg16': vgg_preprocess_input,
                }

model_batch_size = {'densenet169':32,
              'densenet201':30,
              'densenet121':32,                    
             'resnet101':32,
              'xception':32,
              'resnet101v2':32,
              'inception_resnet_v2':32,
              'resnet152v2':32,
              'vgg16':32,
                }

