import os
import tensorflow as tf
import numpy as np
import matplotlib.cm as cm
from werkzeug.utils import secure_filename
import cv2



def save_and_display_gradcam_plusplus(img_path, cam_plusplus, cam_path="grad_cam_plus.jpg", alpha=0.4):
    file_namePlus = os.path.basename(img_path)
    img = tf.keras.preprocessing.image.load_img(img_path)
    img = tf.keras.preprocessing.image.img_to_array(img)
    
    cam_plusplus = cv2.resize(cam_plusplus, (img.shape[1], img.shape[0]))
    cam_plusplus = np.maximum(cam_plusplus, 0)
    cam_plusplus = (cam_plusplus - cam_plusplus.min()) / (cam_plusplus.max() - cam_plusplus.min())  # Normalize to [0, 1]
    
    # Use jet colormap to colorize Grad-CAM++
    jet = cm.get_cmap("jet")
    jet_colors = jet(cam_plusplus)[:, :, :3]
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_colors)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the Grad-CAM++ heatmap on the original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    
    file_namePlus = file_namePlus + "_" + cam_path
    basepath = os.path.dirname(__file__)
    cam_path = os.path.join(basepath, 'uploads', secure_filename(file_namePlus))
    
    # Save the superimposed image
    superimposed_img.save(cam_path)
    
    return file_namePlus

   # Display Grad CAM
   # display(Image(cam_path))

def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = tf.keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def compute_gradcam_plusplus(model, img_array, layer_name):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        class_idx = np.argmax(predictions[0])
        loss = predictions[:, class_idx]

    output = conv_output[0]
    grads = tape.gradient(loss, conv_output)[0]

    guided_grads = tf.cast(output > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads

    weights = tf.reduce_mean(guided_grads, axis=(0, 1))

    alpha_num = guided_grads
    alpha_denom = 2 * tf.reduce_sum(guided_grads, axis=(0, 1))[:, tf.newaxis, tf.newaxis] + tf.constant(1e-7)
    alphas = alpha_num / alpha_denom[..., tf.newaxis]

    deep_linearization_weights = tf.reduce_sum(weights * alphas, axis=(0, 1))
    cam_plusplus = tf.reduce_sum(deep_linearization_weights * output, axis=-1)

    cam_plusplus = cv2.resize(cam_plusplus.numpy(), (img_array.shape[2], img_array.shape[1]))  # Adjust shape for image
    return cam_plusplus
