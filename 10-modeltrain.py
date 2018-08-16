from keras.models import Sequential
from keras.layers import Dense, Dropout

model1008 = Sequential([
    Dense(16, input_shape=(1,), activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1),
])


model1009 = Sequential([
    Dense(16, input_shape=(1,), activation='relu'),
    Dense(32, activation='relu'),
    Dropout(.5),
    Dense(16, activation='relu'),
    Dense(1),
])

model1010 = Sequential([
    Dense(16, input_shape=(1,), activation='relu'),
    Dense(32, activation='relu'),
    Dropout(.5),
    Dense(16, activation='relu'),
    Dense(1),
])
model1008.compile(optimizer='rmsprop', loss='mse') 
model1009.compile(optimizer='adam', loss='mse') 
model1010.compile(optimizer='adam', loss='mse')

ds08 = xfers.where(xfers.SIZE < 10**8).dropna()
ds09 = xfers.where(xfers.SIZE < 10**9).dropna()
ds09 = ds09.where(xfers.SIZE > 10**8).dropna()
ds10 = xfers.where(xfers.SIZE > 10**9).dropna()

np.random.seed(42)
ds08s = ds08.sample(200)
ds09s = ds09.sample(200)
ds10s = ds10.sample(200)

model1008.fit(ds08.SIZE, ds08.rate, epochs=10, batch_size=64)
model1009.fit(ds09.SIZE, ds09.rate, epochs=10, batch_size=64)
model1010.fit(ds10.SIZE, ds10.rate, epochs=10, batch_size=64)

pred08 = model1008.predict(ds08.SIZE)
pred09 = model1009.predict(ds09.SIZE)
pred10 = model1010.predict(ds10.SIZE)

plt.plot(xfers.SIZE, xfers.rate, '.')
plt.plot(ds08.SIZE, pred08, '.')
plt.plot(ds09.SIZE, pred09, '.')
plt.plot(ds10.SIZE, pred10, '.')



