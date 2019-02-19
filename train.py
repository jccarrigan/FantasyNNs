import numpy as np
from keras import models, layers
from keras.utils import normalize

EPOCHS = 80

dataset = np.load('dataset.npy')

print(dataset)

label_column = dataset.shape[1] - 1
input_data = dataset[:, 0: label_column]
targets = dataset[:, label_column]

input_data = normalize(
    input_data.astype('float32'),
    axis=-1,
)

model = models.Sequential()
model.add(layers.Dropout(0.2, input_shape=(input_data.shape[1],)))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1))
print(model.summary())

model.compile(optimizer='adam', loss='mae',
              metrics=['mae'])

history = model.fit(
    x=input_data,
    y=targets,
    validation_split=.25,
    epochs=EPOCHS,
    batch_size=32)

model.save('offensive_player_fp_predict.h5')