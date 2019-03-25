import numpy as np
from keras import models, layers
from keras.utils import normalize

EPOCHS = 200

POINT_INDEX = 2

def main():
    dataset = np.load('dataset.npy')
    x, y = generate(dataset)
    train(x, y)


def train(x, y):
    # input_data = normalize(
    #     x.astype('float32'),
    #     axis=-1,
    # )
    model = models.Sequential()
    model.add(layers.LSTM(64, input_shape=(None, 20)))
    model.add(layers.Dense(1))
    print(model.summary())

    model.compile(optimizer='adam', loss='mae',
                  metrics=['mae'])

    history = model.fit(
        x=x,
        y=y,
        validation_split=.25,
        epochs=EPOCHS,
        batch_size=16)

    model.save('offensive_player_fp_predict.h5')


def generate(dataset):
    samples = []
    targets = []
    for player in dataset:
        prev_week = 0
        data = []
        for week in player:
            if week[-1] < prev_week:
                perm_data, target_vals = perm(data)
                samples.extend(perm_data)
                targets.extend(target_vals)
                data = []
            data.append(week)
            prev_week = week[-1]

    return samples, targets


def perm(data):
    perms = []
    targets = []
    for i, _ in enumerate(data):
        try:
            targets.append(data[i + 1][POINT_INDEX])
        except IndexError:
            continue
        perms.append(data[:i + 1])

    return perms, targets


if __name__ == '__main__':
    main()
