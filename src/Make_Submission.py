from tensorflow import keras
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


im_shape = (128, 128, 3)

def load_image(path, dim=None):
    im = Image.open(path)
    if dim is not None:
        im = im.resize(dim,resample=Image.LANCZOS)
    return np.array(im)


# test
test_data = pd.read_csv("../data/test.csv")

images = np.empty(shape=(len(test_data.index), im_shape[0], im_shape[1], 3))
for i, row in tqdm(test_data.iterrows()):
    im = load_image("../data/test_images/" + str(row.id_code) + ".png", (im_shape[0], im_shape[1]))
    images[i] = im
        
test_images = images


# Recreate the exact same model, including weights and optimizer.
model = keras.models.load_model('basic.h5')


predictions = model.predict(test_images)
# create submission
test_data.reset_index()
with open("submissions.csv", "w") as f:
    f.write("id_code,diagnosis\n")
    for i in range(len(predictions)):
        f.write(test_data.id_code[i] + "," + str(np.argmax(predictions[i])) + "\n")


a = pd.read_csv("submissions.csv")


a.diagnosis.value_counts()
