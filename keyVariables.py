import random
from os import listdir

PATH='dataset2.0'
CLASS_LABLES=listdir(PATH)
IMAGE_HEIGHT=32
IMAGE_WIDTH=32
SEQUENCE_LENGTH=20
SEED=42
EPOCHS=80
BATCH_SIZE=32
random.seed(SEED)