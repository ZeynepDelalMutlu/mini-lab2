import onnxruntime as ort

# load model
try:
    session = ort.InferenceSession("adv_inception_v3_Opset16.onnx")
        
    for input_node in session.get_inputs():
        print(f"model name: {input_node.name}")
        print(f"image name: {input_node.shape}")
        print(f"data type: {input_node.type}")
    
    print("-" * 30)
    
    for output_node in session.get_outputs():
        print(f"output name: {output_node.name}")
        print(f"output shape: {output_node.shape}")

except Exception as e:
    print(f"Error: {e}")