from keras.applications import inception_v3
from keras.preprocessing import image
from pandas import np
from sklearn.tests.test_base import K

model = inception_v3.InceptionV3()
model_input_layer = model.layers[0].input
model_output_layer = model.layers[-1].output

img = image.load_img("D:/Desktop/pig.jpg", target_size=(299, 299))
original_image = image.img_to_array(img)
hacked_image = np.copy(original_image)

max_change_above = original_image + 0.01
max_change_below = original_image - 0.01

object_type_to_fake = 859

cost_function = model_output_layer[0, object_type_to_fake]
gradient_function = K.gradients(cost_function, model_input_layer)[0]
grab_cost_and_gradients_from_model = K.function([model_input_layer, K.learning_phase()],
                                                [cost_function, gradient_function])

e = 0.007
while cost < 0.60:
    cost, gradients = grab_cost_and_gradients_from_model([hacked_image, 0])
    n = np.sign(gradients)
    hacked_image += n * e
    hacked_image = np.clip(hacked_image, max_change_below, max_change_above)
    hacked_image = np.clip(hacked_image, -1.0, 1.0)
    print("batch:{} Cost: {:.8}%".format(index, cost * 100))
    index += 1
