import tensorflow as tf
slim = tf.contrib.slim
from nets.resized_based_seg import Model
from helper import read_image_and_annotation, next_batch, model_input, create_tf_shuffle_batch_queue
from matplotlib import pyplot as plt
import numpy as np
from libs.training import get_labels_from_annotation_batch

# Mean values for VGG-16
from preprocessing.vgg_preprocessing import _R_MEAN, _G_MEAN, _B_MEAN

validation_filenames = "./VOC_train_val_filenames/test.txt"
training_images_dir = "/home/thalles/VOC2012/JPEGImages/"
training_labels_dir = "/home/thalles/VOC2012/SegmentationClass_1D/"

number_of_classes = 21
class_labels = [class_id for class_id in range(number_of_classes)]

filename_queue = tf.train.string_input_producer([validation_filenames], num_epochs=1)

#input_image_placeholder, input_annotation_placeholder, is_training_placeholder = model_input(with_input_placeholder=True)
is_training_placeholder = model_input(with_input_placeholder=False)

image_size = [384, 384]
batch_images, batch_labels = create_tf_shuffle_batch_queue(filename_queue, training_images_dir, training_labels_dir, image_size=image_size, batch_size=1)

model = Model(batch_images, number_of_classes=21, is_training=False, model_name="FCN")
model_segmentation_logits = model.inference()

print(model_segmentation_logits)

get_labels_from_annotation_batch(batch_labels, class_labels)
pred = tf.argmax(model_segmentation_logits, dimension=3)
probabilities = tf.nn.softmax(model_segmentation_logits)

# Define the accuracy metric: Mean Intersection Over Union
miou, miou_update_op = slim.metrics.streaming_mean_iou(predictions=pred,
                                                       labels=batch_labels,
                                                       num_classes=number_of_classes)

acc, acc_update_op = slim.metrics.streaming_accuracy(predictions=pred,
                                                     labels=batch_labels,
                                                     name="pixel_accuracy")

file = open(validation_filenames, 'r')
images_filenale_list = [line for line in file]

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())

    # Restore variables from disk.
    saver.restore(sess, "./checkpoints/FCNCheckpoints")
    print("Model restored.")

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    step = 0
    try:
        while not coord.should_stop():

            pred_np, probabilities_np, images_batch_np, annotation_batch_np, _, _ = \
                sess.run([pred, probabilities, batch_images, batch_labels, miou_update_op, acc_update_op],
                feed_dict={is_training_placeholder: False})

            cmap = plt.get_cmap('bwr')
            f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True)
            f.set_figheight(4)
            f.set_figwidth(18)

            ax1.imshow(images_batch_np.squeeze()[:, :, 0])
            ax1.set_title('Input Image - Truth')

            probability_graph = ax2.imshow(np.dstack((annotation_batch_np[0].astype(float),) * 3) * 100)
            ax2.set_title('Input Annotation - Truth')

            ax3.imshow(np.squeeze(pred_np))
            probability_graph = ax4.imshow(probabilities_np[0].squeeze()[:, :, 0])

            plt.colorbar(probability_graph)
            plt.show()

            step += 1

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
        miou_np = sess.run(miou)
        accuracy = sess.run(acc)
        print("\tmIOU:", miou_np, "\tPixel Accuracy:", accuracy)

    finally:
        # When done, ask the threads to stop.
        coord.request_stop()