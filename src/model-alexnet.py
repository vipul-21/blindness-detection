import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
from tensorflow.keras.utils import plot_model


# Dataset location
train_csv = "/home/resl/Dev/Datasets/APTOS-2019-Blindness-Detection-Dataset/train.csv"
test_csv = "/home/resl/Dev/Datasets/APTOS-2019-Blindness-Detection-Dataset/test.csv"
train_dir = "/home/resl/Dev/Datasets/APTOS-2019-Blindness-Detection-Dataset/train_images/"
test_dir = "/home/resl/Dev/Datasets/APTOS-2019-Blindness-Detection-Dataset/test_images/"
prep_train_dir = "/home/resl/Dev/Datasets/APTOS-2019-Blindness-Detection-Dataset/train_prep_images/"
prep_test_dir = "/home/resl/Dev/Datasets/APTOS-2019-Blindness-Detection-Dataset/test_prep_images/"

use_preprocessed = True
dump_preprocessed = True

if use_preprocessed and os.path.isdir(prep_train_dir) and os.path.isdir(prep_test_dir):
    print("Using preprocessed images")
    train_dir = prep_train_dir
    test_dir = prep_test_dir
    dump_preprocessed = False
else:
    if not os.path.isdir(prep_train_dir):
        os.mkdir(prep_train_dir)
    if not os.path.isdir(prep_test_dir):
        os.mkdir(prep_test_dir)


# Dataset Analysis
train_data = pd.read_csv(train_csv)
test_data = pd.read_csv(test_csv)
class_labels = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

print("Training Size: ", len(train_data))
print("Testing Size: ", len(test_data))

plt.suptitle("Class Distribution")
plt.bar(range(5), train_data.diagnosis.value_counts())
plt.xticks(range(5), class_labels, rotation=50)
plt.ylabel("Samples")
plt.show()


# Some Visualizations
im_shape = (128, 128, 3)


def load_image(path, dim=None):
    im = Image.open(path)
    if dim is not None:
        im = im.resize(dim,resample=Image.LANCZOS)
    return im


num_classes = len(class_labels)
num_samples = 5
figure = plt.figure()

for i, label in enumerate(class_labels):
    samples = train_data.loc[train_data.diagnosis==i, 'id_code'].sample(num_samples,).reset_index(drop=True)
    for j in range(num_samples):
        im = load_image(train_dir + str(samples[j]) + ".png", (im_shape[0], im_shape[1]))
        fig_ax = figure.add_subplot(num_classes, num_classes, i+5*j+1)
        fig_ax.set_xticks([], [])
        fig_ax.set_yticks([], [])
        if j==0:
            fig_ax.set_title(class_labels[i])
        fig_ax.imshow(im)

plt.show()


images = np.empty(shape=(len(train_data.index), im_shape[0], im_shape[1], 3))
labels = np.empty(shape=(len(train_data.index),))
for i, row in tqdm(train_data.iterrows()):
    im = load_image(train_dir + str(row.id_code) + ".png", (im_shape[0], im_shape[1]))
    images[i] = np.array(im)
    labels[i] = row.diagnosis
    if dump_preprocessed:
        im.save(prep_train_dir + str(row.id_code) + ".png")

validation_images = images[:int(len(images)*0.2)]
validation_labels = labels[:int(len(images)*0.2)]
train_images = images[int(len(images)*0.2):]
train_labels = labels[int(len(images)*0.2):]
del(images)

print(train_images.shape, train_labels.shape)
print(validation_images.shape, validation_labels.shape)

# Time (original+saving preprocessed) : 6 min
# Time (preprocessed): 2 s (Awesome!!)


model = keras.Sequential([
    keras.layers.Conv2D(
        input_shape= im_shape,
        filters=48,
        kernel_size=11,
        padding="same",
        strides=4,
        activation=tf.nn.relu,
        name="conv-1",
    ),
    keras.layers.MaxPool2D(
        pool_size=(3,3),
        strides=1,
        padding="same",
        name="max-pool-1",
    ),
    keras.layers.Conv2D(
        filters=128,
        kernel_size=5,
        padding="same",
        strides=4,
        activation=tf.nn.relu,
        name="conv-2",
    ),
    keras.layers.MaxPool2D(
        pool_size=(3,3),
        strides=1,
        padding="same",
        name="maxpool-2",
    ),
    keras.layers.Conv2D(
        filters=192,
        kernel_size=3,
        # padding="same",
        # strides=4,
        activation=tf.nn.relu,
        name="conv-3",
    ),
    keras.layers.Conv2D(
        filters=192,
        kernel_size=3,
        # padding="same",
        # strides=4,
        activation=tf.nn.relu,
        name="conv-4",
    ),
    keras.layers.Conv2D(
        filters=128,
        kernel_size=3,
        # padding="same",
        # strides=4,
        activation=tf.nn.relu,
        name="conv-5",
    ),
    keras.layers.Flatten(),
    keras.layers.Dense(2048, activation=tf.nn.relu),
    keras.layers.Dense(2048, activation=tf.nn.relu),
    keras.layers.Dense(5, activation=tf.nn.softmax)
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.build()
model.summary()
plot_model(model, to_file="model.png", show_shapes=True)


model.fit(train_images, train_labels, epochs=20)
model.evaluate(validation_images, validation_labels)


# # test
# images = np.empty(shape=(len(test_data.index), im_shape[0], im_shape[1], 3))
# for i, row in tqdm(test_data.iterrows()):
#     im = load_image(test_dir + str(row.id_code) + ".png", (im_shape[0], im_shape[1]))
#     images[i] = np.array(im)
#     if dump_preprocessed:
#         im.save(prep_test_dir + str(row.id_code) + ".png")
#
# test_images = images
# # Time:
# # Original + Dumping: 1 min
# # Preprocessed: 1 s


# predictions = model.predict(test_images)


# # create submission
# test_data.reset_index()
# with open("submissions.csv", "w") as f:
#     f.write("id_code, diagnosis\n")
#     for i in range(len(predictions)):
#         f.write(test_data.id_code[i] + "," + str(np.argmax(predictions[i])) + "\n")


# test_data.id_code[0]


# model.save("basic.h5")

