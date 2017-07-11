from __future__ import division
import os
import tensorflow as tf
import numpy as np
from nets.resized_based_seg import Model
from libs.training import get_valid_logits_and_labels
from helper import create_tf_shuffle_batch_queue, next_batch, get_filenames_list, model_input
from libs.training import get_labels_from_annotation_batch
from time import gmtime, strftime

slim = tf.contrib.slim

class_semantic_labels = {
    0: "background",
    1: "aeroplane",
    2: "bicycle",
    3: "bird",
    4: "boat",
    5: "bottle",
    6: "bus",
    7: "car",
    8: "cat",
    9: "chair",
    10: "cow",
    11: "diningtable",
    12: "dog",
    13: "horse",
    14: "motorbike",
    15: "person",
    16: "pottedplant",
    17: "sheep",
    18: "sofa",
    19: "train",
    20: "tvmonitor",
    255: "undefined/don't care"
}

number_of_classes = 21
class_labels = [v for v in range((number_of_classes + 1))]
class_labels[-1] = 255

training_filenames = "./VOC_train_val_filenames/train.txt"
validation_filenames = "./VOC_train_val_filenames/val.txt"
training_images_dir = "/home/thalles/VOC2012/JPEGImages/"
training_labels_dir = "/home/thalles/VOC2012/SegmentationClass_1D/"

checkpoints_dir = '/home/thalles/image-segmentation/vgg'
log_folder = '/home/thalles/log_folder'
vgg_checkpoint_path = os.path.join(checkpoints_dir, 'vgg_16.ckpt')


def model_loss(upsampled_by_factor_16_logits, labels, class_labels, resize_input=False):

    if resize_input:
        valid_labels_batch_tensor, valid_logits_batch_tensor = get_valid_logits_and_labels(annotation_batch_tensor=labels,
                                                                                           logits_batch_tensor=upsampled_by_factor_16_logits,
                                                                                           class_labels=class_labels)

        cross_entropies = tf.nn.softmax_cross_entropy_with_logits(logits=valid_logits_batch_tensor,
                                                                  labels=valid_labels_batch_tensor)
    else:
        valid_labels_batch_tensor = get_labels_from_annotation_batch(labels, class_labels)

        cross_entropies = tf.nn.softmax_cross_entropy_with_logits(logits=upsampled_by_factor_16_logits,
                                                              labels=valid_labels_batch_tensor)

    cross_entropy_mean = tf.reduce_mean(cross_entropies)

    # Add summary op for the loss -- to be able to see it in tensorboard.
    tf.summary.scalar('cross_entropy_loss', cross_entropy_mean)

    # Tensor to get the final prediction for each pixel -- pay
    # attention that we don't need softmax in this case because
    # we only need the final decision. If we also need the respective
    # probabilities we will have to apply softmax.
    pred = tf.argmax(upsampled_by_factor_16_logits, dimension=3)
    probabilities = tf.nn.softmax(upsampled_by_factor_16_logits)

    return cross_entropy_mean, pred, probabilities


def model_optimizer(cross_entropy_sum, learning_rate):
    # Here we define an optimizer and put all the variables
    # that will be created under a namespace of 'adam_vars'.
    # This is done so that we can easily access them later.
    # Those variables are used by adam optimizer and are not
    # related to variables of the vgg model.
    with tf.variable_scope("adam_vars"):
        train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy_sum)
    return train_step


image_size = [384, 384]
batch_size = 1
learning_rate = 0.00000001
total_epochs = 80

filename_queue = tf.train.string_input_producer([training_filenames], num_epochs=total_epochs)

batch_images, batch_labels = create_tf_shuffle_batch_queue(filename_queue, training_images_dir, training_labels_dir, image_size=image_size, batch_size=1)

#batch_images, batch_labels, is_training_placeholder = model_input(with_input_placeholder=True)
is_training_placeholder = model_input(with_input_placeholder=False) # for shuffle batch

model = Model(batch_images, number_of_classes=number_of_classes, is_training=is_training_placeholder, model_name="FCN")
logits = model.inference()

mean_cross_entropy, pred, probabilities = model_loss(logits, batch_labels, class_labels)

train_step = model_optimizer(mean_cross_entropy, learning_rate=learning_rate)

# Define the accuracy metric: Mean Intersection Over Union
miou, miou_update_op = slim.metrics.streaming_mean_iou(predictions=pred,
                                                       labels=batch_labels,
                                                       num_classes=number_of_classes)

acc, acc_update_op = slim.metrics.streaming_accuracy(predictions=pred,
                                                     labels=batch_labels,
                                                     name="pixel_accuracy")


# get all segmentation model vars, these are the variables we create to perform
# the segmentation upsampling layers
model_variables = [var.op.name for var in slim.get_variables(scope="seg_vars")]

# Now we define a function that will load the weights from VGG checkpoint
# into our variables when we call it. We exclude the weights from the last layer
# which is responsible for class predictions. We do this because
# we will have different number of classes to predict and we can't
# use the old ones as an initialization.
exclude_vars = model_variables + ['vgg_16/fc8', 'adam_vars']
vgg_except_fc8_weights = slim.get_variables_to_restore(exclude=exclude_vars)

# Here we get variables that belong to the last layer of network.
# As we saw, the number of classes that VGG was originally trained on
# is different from ours -- in our case it is only 2 classes.
vgg_fc8_weights = slim.get_variables_to_restore(include=['vgg_16/fc8'])

adam_optimizer_variables = slim.get_variables_to_restore(include=['adam_vars'])

# get the segmentation upsampling variables to be initialized
model_variables = slim.get_variables(scope="seg_vars")

# Put all summary ops into one op. Produces string when you run it.
merged_summary_op = tf.summary.merge_all()

# Create the summary writer -- to write all the logs
# into a specified file. This file can be later read
# by tensorboard.
summary_string_writer = tf.summary.FileWriter(log_folder)

# Create the log folder if doesn't exist yet
if not os.path.exists(log_folder):
    os.makedirs(log_folder)

# Create an OP that performs the initialization of
# the VGG net variables.
read_vgg_weights_except_fc8_func = slim.assign_from_checkpoint_fn(
    vgg_checkpoint_path,
    vgg_except_fc8_weights)

# Initializer for new fc8 weights -- for two classes.
vgg_fc8_weights_initializer = tf.variables_initializer(vgg_fc8_weights)

# Initializer for adam variables
optimization_variables_initializer = tf.variables_initializer(adam_optimizer_variables)

model_vars = tf.variables_initializer(model_variables)

# Create a saver.
saver = tf.train.Saver()

training_filenames_list = get_filenames_list(training_filenames)
validation_filenames_list = get_filenames_list(validation_filenames)

saver = tf.train.Saver()

# print("Start training at:", strftime("%Y-%m-%d %H:%M:%S", gmtime()))
#
# with tf.Session() as sess:
#     # Run the initializers.
#     read_vgg_weights_except_fc8_func(sess)
#     sess.run(vgg_fc8_weights_initializer)
#     sess.run(optimization_variables_initializer)
#     sess.run(tf.local_variables_initializer())
#     sess.run(model_vars)
#     step = 0
#
#     for epoch in range(total_epochs):
#
#         train_images_generator = next_batch(training_images_dir, training_labels_dir, training_filenames_list,
#                                             random_crop=False)
#
#
#         for images_batch_np, annotation_batch_np in train_images_generator:
#
#             _, train_loss, summary_string = sess.run([train_step, mean_cross_entropy, merged_summary_op],
#                                      feed_dict={batch_images: images_batch_np,
#                                      batch_labels: annotation_batch_np,
#                                      is_training_placeholder: True})
#
#             if step % 200 == 0:
#                 summary_string_writer.add_summary(summary_string, step)
#
#             if step % 800 == 0:
#
#                 val_images_generator = next_batch(training_images_dir, training_labels_dir, validation_filenames_list,
#                                                   random_crop=False)
#                 total_val_loss = 0.0
#
#                 for images_batch_val_np, annotation_batch_val_np in val_images_generator:
#
#                     predictions_np, probabilities_np, val_loss, _, _ = \
#                         sess.run([pred, probabilities, mean_cross_entropy, miou_update_op, acc_update_op],
#                                  feed_dict={batch_images: images_batch_val_np,
#                                             batch_labels: annotation_batch_val_np,
#                                             is_training_placeholder: False})
#                     total_val_loss += val_loss
#
#                 # Calculate validation miou and accuracy
#                 miou_np = sess.run(miou)
#                 accuracy = sess.run(acc)
#
#                 print("Epoch {0}/{1}".format(epoch+1, total_epochs), "\tTrain step:", step, "\tTraing Loss:", train_loss, "\tValidation loss:", total_val_loss/len(validation_filenames_list), "\tVal mIOU:", miou_np, "\tVal Pixel Accuracy:", accuracy)
#
#                 # at the end of validation, shuffle the validation image names list
#                 np.random.shuffle(validation_filenames_list)
#
#             step += 1
#
#         # at the end of an epoch, shuffle the training dataset image names
#         np.random.shuffle(training_filenames_list)
#
#         # save the model's checkpoints
#         saver.save(sess, './checkpoints/FCNCheckpoints')
#
#     miou_np = sess.run(miou)
#     accuracy = sess.run(acc)
#     summary_string_writer.close()
#
#     print("\tFinal val mIOU:", miou_np, "\tFinal val Pixel Accuracy:", accuracy)
#
# print("End training at:", strftime("%Y-%m-%d %H:%M:%S", gmtime()))


# Training loop for be used when the batches are generated by the create_tf_shuffle_batch_queue() function
with tf.Session() as sess:
    # Run the initializers.
    read_vgg_weights_except_fc8_func(sess)
    sess.run(vgg_fc8_weights_initializer)
    sess.run(optimization_variables_initializer)
    sess.run(tf.local_variables_initializer())
    sess.run(model_vars)

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    step = 0
    try:
        while not coord.should_stop():

            _, train_loss, _, _ = sess.run([train_step, mean_cross_entropy, miou_update_op, acc_update_op],
                                           feed_dict={is_training_placeholder: True})

            if step % 1000 == 0:

                pred_np, probabilities_np = sess.run([pred, probabilities],
                                                    feed_dict={is_training_placeholder: False})

                miou_np = sess.run(miou)
                accuracy = sess.run(acc)

                pred_annotation = np.expand_dims(pred_np[0], axis=2).astype(float)
                print("Train step:", step, "\tTraing Loss:", train_loss, "\tmIOU:", miou_np, "\tAccuracy:", accuracy)
                saver.save(sess, './checkpoints/FCNCheckpoints')

            step += 1

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
        saver.save(sess, './checkpoints/FCNCheckpoints')

    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    summary_string_writer.close()