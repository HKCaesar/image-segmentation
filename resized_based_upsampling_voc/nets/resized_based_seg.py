from nets import vgg
import tensorflow as tf
from helper import bilinear_upsample_weights
slim = tf.contrib.slim
from preprocessing.vgg_preprocessing import (_mean_image_subtraction, _R_MEAN, _G_MEAN, _B_MEAN)

class Model:
    def __init__(self, processed_images, number_of_classes=21, is_training=True, model_name="FCN"):
        if model_name == "FCN":
            self.logits = self.FCNModel(processed_images, number_of_classes, is_training=is_training)
        elif model_name == "RESSeg":
            self.logits = self.RESSeg(processed_images, number_of_classes, is_training=is_training)

    def inference(self):
        return self.logits

    def RESSeg(self, processed_images, number_of_classes=21, is_training=True):

        with slim.arg_scope(vgg.vgg_arg_scope()):
            _, end_points = vgg.vgg_16(processed_images,
                                             num_classes=number_of_classes,
                                             is_training=is_training,
                                             spatial_squeeze=False,
                                             scope='vgg_16')

        input_shape = tf.shape(processed_images)

        #vgg_fc6_feature = end_points['vgg_16/fc6']

        # get the vggs pool5 feature map
        pool5_feature_map = end_points['vgg_16/pool5']
        pool5_feature_shape = tf.shape(pool5_feature_map)

        pool4_feature_map = end_points['vgg_16/pool4']
        pool4_feature_shape = tf.shape(pool4_feature_map)

        pool3_feature_map = end_points['vgg_16/pool3']
        pool3_feature_shape = tf.shape(pool3_feature_map)

        pool2_feature_map = end_points['vgg_16/pool2']
        pool2_feature_shape = tf.shape(pool2_feature_map)

        pool1_feature_map = end_points['vgg_16/pool1']
        pool1_feature_shape = tf.shape(pool1_feature_map)

        with tf.variable_scope("seg_vars"):

            # Merging Pooling 4 layer
            pool5_feature_resized = tf.image.resize_images(pool5_feature_map,
                                                           [pool4_feature_shape[1], pool4_feature_shape[2]], method=0)
            pool5_feature_conv = slim.conv2d(pool5_feature_resized,
                                             number_of_classes,
                                             [1, 1],
                                             activation_fn=None)

            pool4_feature_conv = slim.conv2d(pool4_feature_map,
                                       number_of_classes,
                                       [1, 1],
                                       activation_fn=None,
                                       normalizer_fn=None,
                                       weights_initializer=tf.zeros_initializer,
                                       scope='seg_vars/pool4')

            pool5_conv_plus_poo4_conv = pool5_feature_conv + pool4_feature_conv

            # Merging Pooling 3 layer
            pool4_feature_resized = tf.image.resize_images(pool5_conv_plus_poo4_conv,
                                                           (pool3_feature_shape[1], pool3_feature_shape[2]), method=0)

            pool4_feature_conv = slim.conv2d(pool4_feature_resized,
                                             number_of_classes,
                                             [1, 1],
                                             activation_fn=None)


            pool3_feature_conv = slim.conv2d(pool3_feature_map,
                                       number_of_classes,
                                       [1, 1],
                                       activation_fn=None,
                                       normalizer_fn=None,
                                       weights_initializer=tf.zeros_initializer,
                                       scope='seg_vars/pool3')

            pool4_conv_plus_pool3_conv = pool4_feature_conv + pool3_feature_conv

            # Merging Pooling 2 layer
            pool3_feature_resized = tf.image.resize_images(pool4_conv_plus_pool3_conv,
                                                           (pool2_feature_shape[1], pool2_feature_shape[2]), method=0)

            pool3_feature_conv = slim.conv2d(pool3_feature_resized,
                                             number_of_classes,
                                             [1, 1],
                                             activation_fn=None)

            pool2_feature_conv = slim.conv2d(pool2_feature_map,
                                   number_of_classes,
                                   [1, 1],
                                   activation_fn=None,
                                   normalizer_fn=None,
                                   weights_initializer=tf.zeros_initializer)

            pool3_conv_plus_pool2_conv = pool3_feature_conv + pool2_feature_conv

            # Merging Pooling1 layer
            pool2_feature_resized = tf.image.resize_images(pool3_conv_plus_pool2_conv,
                                                           (pool1_feature_shape[1], pool1_feature_shape[2]), method=0)

            pool2_feature_conv = slim.conv2d(pool2_feature_resized,
                                             number_of_classes,
                                             [1, 1],
                                             activation_fn=None)

            pool1_feature_conv = slim.conv2d(pool1_feature_map,
                                   number_of_classes,
                                   [1, 1],
                                   activation_fn=None,
                                   normalizer_fn=None,
                                   weights_initializer=tf.zeros_initializer)

            pool2_conv_plus_pool1_conv = pool2_feature_conv + pool1_feature_conv

            # Resize to original image shape
            orignal_image_shape = tf.image.resize_images(pool2_conv_plus_pool1_conv,
                                                           (input_shape[1], input_shape[2]), method=0)

            orignal_image_logits = slim.conv2d(orignal_image_shape,
                                             number_of_classes,
                                             [1, 1],
                                             activation_fn=None)

            return orignal_image_logits

    def FCNModel(self, processed_images, number_of_classes=21, is_training=True):

        upsample_filter_factor_2_np = bilinear_upsample_weights(factor=2,
                                                                number_of_classes=number_of_classes)

        upsample_filter_factor_8_np = bilinear_upsample_weights(factor=4,
                                                                number_of_classes=number_of_classes)
        upsample_filter_factor_2_tensor = tf.constant(upsample_filter_factor_2_np)
        upsample_filter_factor_8_tensor = tf.constant(upsample_filter_factor_8_np)

        processed_images = processed_images - [_R_MEAN, _G_MEAN, _B_MEAN]

        with slim.arg_scope(vgg.vgg_arg_scope()):
            _, end_points = vgg.vgg_16(processed_images,
                                       num_classes=number_of_classes,
                                       is_training=is_training,
                                       spatial_squeeze=False,
                                       fc_conv_padding='SAME')

        # get the vggs pool5 feature map, this way we do not use the last layer of the vgg net therefore making the net faster
        pool5_feature_map = end_points['vgg_16/pool5']

        pool5_logits = slim.conv2d(pool5_feature_map,
                                   number_of_classes,
                                   [1, 1],
                                   activation_fn=None,
                                   normalizer_fn=None,
                                   scope="seg_vars/pool5",
                                   weights_initializer=tf.zeros_initializer)  # Out: # (1, 22, 30, 2)

        pool5_layer_logits_shape = tf.shape(pool5_logits)

        # Calculate the ouput size of the upsampled tensor
        last_layer_upsampled_by_factor_2_logits_shape = tf.stack([
            pool5_layer_logits_shape[0],
            pool5_layer_logits_shape[1] * 2,
            pool5_layer_logits_shape[2] * 2,
            pool5_layer_logits_shape[3]
        ])

        # Perform the upsampling
        last_layer_upsampled_by_factor_2_logits = tf.nn.conv2d_transpose(pool5_logits,
                                                                         upsample_filter_factor_2_tensor,
                                                                         output_shape=last_layer_upsampled_by_factor_2_logits_shape,
                                                                         strides=[1, 2, 2, 1])

        ## Adding the skip here for FCN-16s model

        # We created vgg in the fcn_8s name scope -- so
        # all the vgg endpoints now are prepended with fcn_8s name
        pool4_features = end_points['vgg_16/pool4']

        # We zero initialize the weights to start training with the same
        # accuracy that we ended training FCN-32s

        pool4_logits = slim.conv2d(pool4_features,
                                   number_of_classes,
                                   [1, 1],
                                   activation_fn=None,
                                   normalizer_fn=None,
                                   weights_initializer=tf.zeros_initializer,
                                   scope='seg_vars/pool4')

        fused_last_layer_and_pool4_logits = pool4_logits + last_layer_upsampled_by_factor_2_logits

        fused_last_layer_and_pool4_logits_shape = tf.shape(fused_last_layer_and_pool4_logits)

        # Calculate the ouput size of the upsampled tensor
        fused_last_layer_and_pool4_upsampled_by_factor_2_logits_shape = tf.stack([
            fused_last_layer_and_pool4_logits_shape[0],
            fused_last_layer_and_pool4_logits_shape[1] * 2,
            fused_last_layer_and_pool4_logits_shape[2] * 2,
            fused_last_layer_and_pool4_logits_shape[3]
        ])

        # Perform the upsampling
        fused_last_layer_and_pool4_upsampled_by_factor_2_logits = tf.nn.conv2d_transpose(fused_last_layer_and_pool4_logits,
                                                                                         upsample_filter_factor_2_tensor,
                                                                                         output_shape=fused_last_layer_and_pool4_upsampled_by_factor_2_logits_shape,
                                                                                         strides=[1, 2, 2, 1])

        ## Adding the skip here for FCN-8s model
        pool3_features = end_points['vgg_16/pool3']

        # We zero initialize the weights to start training with the same
        # accuracy that we ended training FCN-32s

        pool3_logits = slim.conv2d(pool3_features,
                                   number_of_classes,
                                   [1, 1],
                                   activation_fn=None,
                                   normalizer_fn=None,
                                   weights_initializer=tf.zeros_initializer,
                                   scope='seg_vars/pool3')

        fused_last_layer_and_pool4_logits_and_pool_3_logits = pool3_logits + \
                                                              fused_last_layer_and_pool4_upsampled_by_factor_2_logits

        fused_last_layer_and_pool4_logits_and_pool_3_logits_shape = tf.shape(
            fused_last_layer_and_pool4_logits_and_pool_3_logits)

        # Calculate the ouput size of the upsampled tensor
        fused_last_layer_and_pool4_logits_and_pool_3_upsampled_by_factor_8_logits_shape = tf.stack([
            fused_last_layer_and_pool4_logits_and_pool_3_logits_shape[0],
            fused_last_layer_and_pool4_logits_and_pool_3_logits_shape[1] * 8,
            fused_last_layer_and_pool4_logits_and_pool_3_logits_shape[2] * 8,
            fused_last_layer_and_pool4_logits_and_pool_3_logits_shape[3]
        ])

        # Perform the upsampling
        fused_last_layer_and_pool4_logits_and_pool_3_upsampled_by_factor_8_logits = tf.nn.conv2d_transpose(
            fused_last_layer_and_pool4_logits_and_pool_3_logits,
            upsample_filter_factor_8_tensor,
            output_shape=fused_last_layer_and_pool4_logits_and_pool_3_upsampled_by_factor_8_logits_shape,
            strides=[1, 8, 8, 1])

        return fused_last_layer_and_pool4_logits_and_pool_3_upsampled_by_factor_8_logits
