import numpy as np
import tflite
import capture_photo

from PIL import Image

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="money-model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# get input model shape.
_, height, width, _ = input_details[0]['shape']

# Load an image and reshape with the input model shape in order to be classified.
image = Image.open(capture_photo.capture_key_pressed()).resize((width, height))

# classify the image.

input_tensor = np.array(np.expand_dims(image, 0), dtype=np.float32)
input_index = interpreter.get_input_details()[0]["index"]
interpreter.set_tensor(input_index, input_tensor)
interpreter.invoke()
output_details = interpreter.get_output_details()

output_data = interpreter.get_tensor(output_details[0]['index'])

prediction = np.squeeze(output_data)

# prediction without score calculated
class_names = {0: '10', 1: '20', 2: '5', 3: '50'}
highest_pred_loc = np.argmax(prediction)
print("This image most likely belongs to {} dt".format(class_names[highest_pred_loc]))

# prediction with score calculated
'''score = tf.nn.softmax(output_data[0])
print("This image most likely belongs to {} dt with a {:.2f} percent confidence.".format(class_names[np.argmax(score)],
                                                                                         100 * np.max(score)))
'''