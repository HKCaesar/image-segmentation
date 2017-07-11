import tensorflow as tf
slim = tf.contrib.slim
from libs.scale_input_image import scale_randomly_image_with_annotation_with_fixed_size_output
# Load the mean pixel values and the function
# that performs the subtraction from each pixel
from preprocessing.vgg_preprocessing import (_mean_image_subtraction, _R_MEAN, _G_MEAN, _B_MEAN)
from scipy.misc import imread
import numpy as np


def model_input(with_input_placeholder=False):
    input_image_placeholder = tf.placeholder(tf.float32, shape=(None, None, None, 3))
    input_anotation_placeholder = tf.placeholder(tf.float32, shape=(None, None, None))
    is_training_placeholder = tf.placeholder(tf.bool)

    if with_input_placeholder:
        return input_image_placeholder, input_anotation_placeholder, is_training_placeholder
    else:
        return is_training_placeholder

def create_tf_shuffle_batch_queue(filename_queue, images_dir, labels_dir, image_size=[384, 384], batch_size=1):
    reader = tf.TextLineReader()
    key, img_filename = reader.read(filename_queue)

    # read the input and annotation images
    image_tensor = tf.read_file(images_dir + img_filename + ".jpg")
    annotation_tensor = tf.read_file(labels_dir + img_filename + ".png")

    image_tensor = tf.image.decode_jpeg(image_tensor, channels=3)
    annotation_tensor = tf.image.decode_png(annotation_tensor, channels=1)

    resized_image, resized_annotation = scale_randomly_image_with_annotation_with_fixed_size_output(image_tensor,
                                                                                                    annotation_tensor,
                                                                                                    image_size)

    resized_annotation = tf.squeeze(resized_annotation)
    batch_images, batch_labels = tf.train.shuffle_batch([resized_image, resized_annotation],
                                                        batch_size=batch_size,
                                                        capacity=200,
                                                        num_threads=2,
                                                        min_after_dequeue=100)
    return batch_images, batch_labels


def read_image_and_annotation(images_dir, annotations_dir, image_name):
    # read the input and annotation images
    image_np = imread(images_dir + image_name.strip() + ".jpg")
    # Subtract the mean pixel value from each pixel
    image_np = image_np - [_R_MEAN, _G_MEAN, _B_MEAN]

    annotation_np = imread(annotations_dir + image_name.strip() + ".png")
    assert (image_np.dtype) == "float"
    assert (annotation_np.dtype) == "uint8"
    return image_np, annotation_np


def random_crop(image_np, annotation_np, crop_width=246, crop_height=112):
    """
    image_np: rgb image shape (H,W,3)
    annotation_np: 1D image shape (H,W,1)
    crop_size: integer
    """
    image_h = image_np.shape[0]
    image_w = image_np.shape[1]

    random_x = np.random.randint(0,
                                 image_w - crop_width + 1)  # Return random integers from low (inclusive) to high (exclusive).
    random_y = np.random.randint(0,
                                 image_h - crop_height + 1)  # Return random integers from low (inclusive) to high (exclusive).

    offset_x = random_x + crop_width
    offset_y = random_y + crop_height

    return image_np[random_y:offset_y, random_x:offset_x:], annotation_np[random_y:offset_y, random_x:offset_x:]


def next_batch(train_images_dir, train_annotations_dir, image_filenames_list, random_crop=False,
               crop_width=246, crop_height=112, batch_size=5):
    """
    return
    batch_images shape (batch_size, crop_size, crop_size, 3)
    batch_annotations shape (batch_size, crop_size, crop_size)
    """
    batch_images = None
    batch_labels = None

    for image_counter, image_name in enumerate(image_filenames_list, 1):

        image_np, annotation_np = read_image_and_annotation(train_images_dir, train_annotations_dir, image_name.strip())

        if random_crop:
            image_np, annotation_np = random_crop(image_np, annotation_np, crop_width, crop_height)
        else:
            # if the random_crop option is False, then batch size is always 1 because
            # the dataset might have different image sizes
            batch_size = 1

        image_np = np.expand_dims(image_np, axis=0)
        annotation_np = np.expand_dims(annotation_np, axis=0)

        if batch_images is None:
            batch_images = image_np
            batch_labels = annotation_np
        else:
            batch_images = np.concatenate((batch_images, image_np), axis=0)
            batch_labels = np.concatenate((batch_labels, annotation_np), axis=0)

        if batch_images.shape[0] == batch_size:
            yield batch_images, batch_labels
            batch_images = None
            batch_labels = None


def get_filenames_list(training_filenames):
    file = open(training_filenames, 'r')
    training_filenames_list = [line for line in file]
    file.close()
    return training_filenames_list


import numpy as np


def get_kernel_size(factor):
    """
    Find the kernel size given the desired factor of upsampling.
    """
    return 2 * factor - factor % 2


def upsample_filt(size):
    """
    Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
    """
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)


def bilinear_upsample_weights(factor, number_of_classes):
    """
    Create weights matrix for transposed convolution with bilinear filter
    initialization.
    """

    filter_size = get_kernel_size(factor)

    weights = np.zeros((filter_size,
                        filter_size,
                        number_of_classes,
                        number_of_classes), dtype=np.float32)

    upsample_kernel = upsample_filt(filter_size)

    for i in range(number_of_classes):
        weights[:, :, i, i] = upsample_kernel

    return weights