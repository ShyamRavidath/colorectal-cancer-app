import h5py
from tensorflow.keras.models import model_from_json

# Load the model configuration
with h5py.File('Model.h5', 'r') as f:
    model_config = f.attrs.get('model_config')
    if model_config is None:
        raise ValueError("No model configuration found in the file.")

    # Check if model_config is bytes or string
    if isinstance(model_config, bytes):
        model_config = model_config.decode('utf-8')  # Decode if it's bytes
    elif isinstance(model_config, str):
        pass  # Already a string, no need to decode
    else:
        raise TypeError(f"Unexpected type for model_config: {type(model_config)}")

    # Replace 'batch_shape' with 'input_shape'
    model_config = model_config.replace('"batch_shape":', '"input_shape":')

# Recreate the model
model = model_from_json(model_config)
model.load_weights('Model.h5')

print("Model successfully loaded!")
