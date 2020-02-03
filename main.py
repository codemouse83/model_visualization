# -*- coding: utf-8 -*- #
# ===== Import Module ==== #
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import keras.backend as K

import matplotlib.pyplot as plt
import numpy as np
import json
import shap

# ===== Config ===== #
class_names = None
model = None
img_model =model = VGG16(weights='imagenet', include_top=True)
all_input,y = shap.datasets.imagenet50()
x = all_input[[5,15]]

# ==== Main Class ===== #
class model_visualization:
    def __init__(self,input,all_input,class_names,model):
        self.input = input
        self.all_input = all_input
        self.model = model
        self.class_names = class_names
        self.img_layers = 7

    def SHAP_img(self):
        def map2layer(x,layer):
            feed_dict = dict(zip([self.model.layers[0].input],[preprocess_input(x.copy())]))
            return K.get_session().run(self.model.layers[layer].input,feed_dict)

        e = shap.GradientExplainer(
            (model.layers[self.img_layers].input,model.layers[-1].output),
            map2layer(self.all_input,self.img_layers),
            local_smoothing=0)

        shap_values,predict = e.shap_values(map2layer(self.input,self.img_layers),ranked_outputs=1)
        index_names = np,vectorize(lambda x:self.class_names[str(x)][1])(indexes)
        return shape_values,predict







# ===== Sample Process ===== #
if __name__ == "__main__":
    url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
    fname = shap.datasets.cache(url)
    with open(fname) as f:
        class_names = json.load(f)

    m_c = model_visualization(x,all_input,class_names,img_model)
    print(m_c.all_input.shape)
    shape_values,predict = m_c.SHAP_img()
    print(values.shape,predict)
