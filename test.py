
import numpy as np
import onnxruntime as ort

img = np.load("./assets/image.npz").reshape([1, 784])
sess_ort = ort.InferenceSession("./output/mnist1.onnx")
res = sess_ort.run(output_names=[output_tensor.name], input_feed={input_tensor.name: img})
print("the expected result is \"7\"")
print("the digit is classified as \"%s\" in ONNXRruntime"%np.argmax(res))

def init():
    sess_ort = ort.InferenceSession("./models/arcfaceresnet100-8.onnx")
    print("시작합니다~")
    print(sess_ort)


init()
