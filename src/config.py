import os

DATA_DIR = "/home/resl/Dev/Datasets/APTOS-2019-Blindness-Detection-Dataset/"

TRAIN_CSV = DATA_DIR + "train.csv"
TEST_CSV = DATA_DIR + "test.csv"

TRAIN_DIR = DATA_DIR + "train_images/"
TEST_DIR = DATA_DIR + "test_images/"

PREPROCESSED_DIR = DATA_DIR + "preprocessed/"
PREPROCESSED_TRAIN_DIR = DATA_DIR + "train_prep_images/"
PREPROCESSED_TEST_DIR = DATA_DIR + "test_prep_images/"

USE_PREPROCESSED = False
DUMP_PREPROCESSED = True

if (
        USE_PREPROCESSED
        and os.path.isdir(PREPROCESSED_TRAIN_DIR)
        and os.path.isdir(PREPROCESSED_TEST_DIR)
):
    TRAIN_DIR = PREPROCESSED_TRAIN_DIR
    TEST_DIR = PREPROCESSED_TEST_DIR
    DUMP_PREPROCESSED = False
else:
    if not os.path.isdir(PREPROCESSED_TRAIN_DIR):
        os.mkdir(PREPROCESSED_TRAIN_DIR)
    if not os.path.isdir(PREPROCESSED_TEST_DIR):
        os.mkdir(PREPROCESSED_TEST_DIR)

VALIDATION_FRACTION = 0.2
PREPARE_SUBMIT = True 

OLD_DATA_DIR = "/home/resl/Dev/Datasets/drd/"

OLD_TRAIN_CSV = OLD_DATA_DIR + "train.csv"
OLD_TEST_CSV = OLD_DATA_DIR + "test.csv"

OLD_TRAIN_DIR = OLD_DATA_DIR + "train_images/"
OLD_TEST_DIR = OLD_DATA_DIR + "test_images/"

OLD_PREPROCESSED_DIR = OLD_DATA_DIR + "preprocessed/"