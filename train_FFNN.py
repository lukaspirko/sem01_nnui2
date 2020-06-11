import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from matplotlib import pyplot as plt

# tvorba trenovaci mnoziny
pocet_dat = 2000
np.random.seed(1234)
vstupy = 2 * np.random.rand(pocet_dat, 3) - 1
cile = np.zeros((pocet_dat, 2))

for ind in range(pocet_dat):
    cile[ind, 0] = max(vstupy[ind]) * vstupy[ind, 1]
    cile[ind, 1] = pow(vstupy[ind, 0], 2) - (vstupy[ind, 1] * vstupy[ind, 2])

# tvorba modelu
model = Sequential()

# tvorba vrstev
# vytvoreni skryte vrstvy
model.add(Dense(36, input_dim=3, activation='tanh'))
# vytvoreni vystupni vrstvy
model.add(Dense(2, activation='linear'))

model.compile(optimizer='rmsprop', loss='mse')
model.summary()

# pocet epoch
epoch = 2000
# velikost davky
batch = 32
# % pouzitych dat na validace
val = 0.15
hist = model.fit(x=vstupy, y=cile, epochs=epoch, batch_size=batch, validation_split=val, verbose=2)

# zobrazeni grafu
# trenovaci - blue
plt.plot(hist.history['loss'])
# validacni - orange
plt.plot(hist.history['val_loss'])
plt.show()

# ulozeni modelu
model.save('model.h5')
