import os
import numpy as np

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import multi_gpu_model

from models import create_auto_encoder
from losses import r_squared, CustomLoss
from params import args


model = create_auto_encoder(filters=args.filters)
model = multi_gpu_model(model, gpus=2)
model.load_weights(args.weights_path)

# Load the initial input frame (frame index 1784 from the test set by default)
x_test = np.load(args.data_path)
x_test = np.expand_dims(x_test[1784, :, :, :], axis=0)

# Output predictions alongside the weights file
output_dir = os.path.dirname(os.path.abspath(args.weights_path))

# Autoregressive prediction loop: each iteration feeds the previous output back in
for i in range(100):
    if i == 0:
        preds = model.predict(x_test)
        del x_test
    else:
        x = np.load(os.path.join(output_dir, str(i) + ".npy"))
        preds = model.predict(x)

    preds = np.asarray(preds)
    np.save(os.path.join(output_dir, str(i + 1) + ".npy"), preds)
    print(preds.shape)
