import numpy as np
import keras.utils as image
from keras_vggface.vggface import VGGFace
from keras_vggface import utils

model = VGGFace(model='vgg16')
# load the image
img = image.load_img(
    './Matthias_Sammer.jpg',
    target_size=(224, 224))

# prepare the image
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = utils.preprocess_input(x, version=1)

# perform prediction
preds = model.predict(x)
print('Predicted:', utils.decode_predictions(preds))