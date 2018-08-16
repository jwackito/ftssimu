from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential([
    Dense(128, input_shape=(1,), activation='elu'),
    Dropout(.15),
    Dense(64, activation='elu'),
    Dropout(.15),
    Dense(32, activation='elu'),
    Dropout(.15),
    Dense(16, activation='elu'),
    Dropout(.15),
    Dense(16, activation='elu'),
    Dropout(.15),
    Dense(32, activation='elu'),
    Dropout(.15),
    Dense(64, activation='elu'),
    Dropout(.15),
    Dense(128, activation='elu'),
    Dense(256, activation='elu'),
    Dropout(.5),
    Dense(1),
])
model.compile(optimizer='rmsprop', loss='hinge') 
np.random.seed(43)
s = xfers.sample(500)

model.fit(s.SIZE, s.rate, epochs=100, batch_size=64)

pred = model.predict(xfers.SIZE)

plt.plot(xfers.SIZE, xfers.rate, '.')
plt.plot(xfers.SIZE, pred, '.')
