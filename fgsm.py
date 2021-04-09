from __future__ import absolute_import, division, print_function, unicode_literals
import warnings

warnings.filterwarnings('ignore')
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['figure.figsize'] = (8, 8)
mpl.rcParams['axes.grid'] = False

# Let's load the pretained MobileNetV2 model and the ImageNet class names.
pretrained_model = tf.keras.applications.MobileNetV2(include_top=True, weights='imagenet')
pretrained_model.trainable = False

# ImageNet labels
decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions


# Helper function to preprocess the image so that it can be inputted in MobileNetV2
# 对图像预处理，使得图像可以输入MobileNetV2模型
def preprocess(image):
    image = tf.cast(x=image, dtype=tf.float32)
    image = image / 255
    image = tf.image.resize(images=image, size=(224, 224))
    image = image[None, ...]
    return image


# Helper function to extract labels from probability vector
# 得到图像真实标签
def get_imagenet_label(probs):
    return decode_predictions(probs, top=1)[0][0]


image_path = tf.keras.utils.get_file(fname='YellowLabradorLooking_new.jpg',
                                     origin='https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
image_raw = tf.io.read_file(filename=image_path)

image = tf.image.decode_image(contents=image_raw)

image = preprocess(image)
image_probs = pretrained_model.predict(image)

plt.figure()
plt.imshow(image[0])
_, image_class, class_confidence = get_imagenet_label(image_probs)
plt.title('{} : {:.2f}% Confidence'.format(image_class, class_confidence * 100))
plt.show()

loss_object = tf.keras.losses.CategoricalCrossentropy()


def create_adversarial_pattern(input_image, input_label):
    with tf.GradientTape() as tape:
        tape.watch(tensor=input_image)
        prediction = pretrained_model(input_image)
        loss = loss_object(input_label, prediction)

    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, input_image)
    # Get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(gradient)  # 符号梯度攻击
    return signed_grad


perturbations = create_adversarial_pattern(image, image_probs)
plt.imshow(perturbations[0])


def display_images(image, description):  # 展示被攻击过的图像
    _, label, confidence = get_imagenet_label(pretrained_model.predict(image))
    plt.figure()
    plt.imshow(image[0])
    plt.title('{} \n {} : {:.2f}% Confidence'.format(description,
                                                     label, confidence * 100))
    plt.show()


epsilons = [0, 0.01, 0.1, 0.15, 0.20]
descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input') for eps in epsilons]

for i, eps in enumerate(epsilons):
    adv_x = image + eps * perturbations
    adv_x = tf.clip_by_value(adv_x, 0, 1)  # 将值限制在[0,1]范围内
    display_images(adv_x, descriptions[i])
