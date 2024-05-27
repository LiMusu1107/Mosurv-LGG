# -*- coding: utf-8 -*-

# Load required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import tensorflow.keras.optimizers as optimizer
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

#%%
# input file
file_input = "LGG_stacked.csv"
# output file
file_output = "LGG_dr.csv"
# checkpoint file
file_checkpoint = "checkpoint/{epoch:02d}_{val_loss:.2f}.hdf5"
# loss image
file_png = "LGG_dr_loss.png"

#%%
# preprocess
print("Reading data \n")
training_data = pd.read_csv(file_input, header = None)
print(f"Dimension of training data: {training_data.shape}")

# split
X_train, X_test, train_ground,valid_ground = train_test_split(
    training_data, 
    training_data, 
    test_size = 0.1, 
    shuffle = True,
    random_state = 2023110400
    )


#%%
# save weights
checkpoint = ModelCheckpoint(
    filepath = file_checkpoint, 
    save_weights_only = True, 
    verbose = 1, 
    monitor = 'val_loss',
    mode = 'min',
    save_best_only = True
    )

# Early stopping
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 5)

# Structure of AE

dim_input = X_train.shape[1]
dim_layer1 = 500
dim_bottleneck = 100


# Encoder 
dim_input_shape = Input(shape = (dim_input,), name="input_layer")
x = Dense(dim_layer1, activation='relu', kernel_initializer = 'uniform', name = 'encoder_layer_1')(dim_input_shape)
encoded = Dense(dim_bottleneck, activation='relu', kernel_initializer = 'uniform', name="bottleneck_layer")(x)

# Decoder
y = Dense(dim_layer1, activation='relu', kernel_initializer = 'uniform', name="decoder_layer_1")(encoded)
decoded = Dense(dim_input, activation='relu', kernel_initializer = 'uniform', name="op_layer")(y)

# Model
autoencoder = Model(inputs = dim_input_shape, outputs = decoded)
autoencoder.summary()

# Hyper-paramters 
lr = 0.001 
batch = 12 
epochs = 500

# Optimizer
opt_adam = optimizer.Adam(learning_rate = lr)

# Compile
autoencoder.compile(optimizer = opt_adam, loss = 'mean_squared_error')

# Train
history = autoencoder.fit(
    X_train,
    X_train, # input and output data
    epochs = epochs,
    batch_size = batch,
    shuffle = True,
    validation_data = (X_test, X_test), # Validation data
    callbacks = [early_stopping, checkpoint]
    )

#%%
# encoder
encoder = Model(inputs = dim_input_shape, outputs = encoded)
# Latent representation
latent_representation = encoder.predict(training_data)
print(f"Dimension of latent representation: {latent_representation.shape}")


# save
df_latent_representation = pd.DataFrame(
    data = latent_representation, 
    index = training_data.index
    )
print(df_latent_representation.iloc[0:10, 0:5])

df_latent_representation.to_csv(file_output, index = False)

#%%
# loss plot
print("Training Loss: ", history.history['loss'][-1])
print("Validation Loss: ", history.history['val_loss'][-1])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'], loc = 'upper right')
plt.savefig(file_png)
plt.close()