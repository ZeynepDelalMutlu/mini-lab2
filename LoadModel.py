import onnxruntime as ort
import numpy as np
from PIL import Image
import time

# 1. Load the model (Create session)
try:
    print("Load model with Python onnxruntime:")
    session = ort.InferenceSession("adv_inception_v3_Opset16.onnx")
    print("Model is loaded!")
except Exception as e:
    print(f"Error while model is loading: {e}")

# 2. Prepare photo (the size is 299x299 ve the format is NCHW)
def prepare_image(img_path):
    img = Image.open(img_path).convert('RGB') # be sure that it is RGB
    img = img.resize((299, 299))
    img_data = np.array(img).astype('float32') / 255.0
    
    # Model is waiting [1, 3, 299, 299] (NCHW)
    img_data = np.transpose(img_data, (2, 0, 1)) # HWC -> CHW
    img_data = np.expand_dims(img_data, axis=0)  # CHW -> NCHW
    return img_data

# 3. Run Prediction
try:
    input_tensor = prepare_image("A-Cat.jpg")

    # netron.app investigation: input 'x', output '875'
    outputs = session.run(["875"], {"x": input_tensor})

    print("1000 prediction processes are started...")
    start_time = time.process_time()    
    for _ in range(1000):
        outputs = session.run(["875"], {"x": input_tensor})
    end_time = time.process_time()

    # Run for a last time to get the results
    # The class with the heighest probability
    outputs = session.run(["875"], {"x": input_tensor})
    total_time = (end_time - start_time) * 1000
    average_time_ms = total_time / 1000
    prediction = np.argmax(outputs[0])
    print(f"Predicted index: {prediction}")
    print(f"Max Logit: {np.max(outputs[0])}")
    print(f"Total time: {(end_time - start_time) * 1000:.2f} ms")
    print(f"Average time: {(end_time - start_time):.2f} ms")

    
except Exception as e:
    print(f"Error runtime: {e}")