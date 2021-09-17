
# ---------模型結構--------------------------------------------------
model = Sequential()

model.add(Conv2D(32, (3, 3), inpur_shape=(150, 150, 3),
          padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

# ---------模型結構--------------------------------------------------

# -----------------------------------------------------------

epochs = 50
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
mosel.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

# -----------------------------------------------------------

model.summary()  # 檢視模型結構


# Callback for loss logging per ephoch
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))


history = LossHistory()

# Callback for early stopping the training
early_stopping-keras.callbacks.EarlyStopping
(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')
