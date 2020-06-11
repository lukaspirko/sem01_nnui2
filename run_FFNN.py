import numpy as np
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt

# nacteni modelu site
model = load_model('model.h5')
model.summary()

# generovani dat
pocet_dat = 30
np.random.seed(123)
vstupy = 2 * np.random.rand(pocet_dat, 3) - 1
cile = np.zeros((pocet_dat, 2))
vystupy = np.zeros((pocet_dat, 2))

for ind in range(pocet_dat):
    # vypocet cilovych hodnot
    cile[ind, 0] = max(vstupy[ind]) * vstupy[ind, 1]
    cile[ind, 1] = pow(vstupy[ind, 0], 2) - (vstupy[ind, 1] * vstupy[ind, 2])

    # vypocet hodnot pomoci neu. site
    vystupy[ind] = model.predict(np.array([[vstupy[ind, 0], vstupy[ind, 1], vstupy[ind, 2]]]))

    # vypis odchylky
    print('\nodchylka X' + str(abs(cile[ind, 0] - vystupy[ind, 0])))
    print('odchylka Y' + str(abs(cile[ind, 1] - vystupy[ind, 1])))

# zobrazeni grafu vysledku
plt.plot(cile[:, 0], cile[:, 1], 'ro')
plt.plot(vystupy[:, 0], vystupy[:, 1], 'bo')
plt.show()
