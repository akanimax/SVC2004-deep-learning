'''
    This cell defines the method required to spawn and return a tensorflow graph for the autoencoder model.
    coded by: Animesh
'''

import tensorflow as tf

data_size = (96, 192, 3)

myGraph = tf.Graph() #create a new graph object

with myGraph.as_default():
    # define the computations of this graph here:
    
    # helper functions to run the model:
    def normalize(ip_tensor, name = "normalization"):
        '''
            function to normalize the input tensor in the range of [-1, 1] 
            @param
            ip_tensor => the tensor to be normalized
            @return => the normalized version of the tensor
        '''
        with tf.name_scope(name): 
            ip_range = tf.reduce_max(ip_tensor) - tf.reduce_min(ip_tensor)
            mean = tf.reduce_mean(ip_tensor)
            return (ip_tensor - mean) / ip_range
    
    

    # placeholder for the input data batch
    inputs = tf.placeholder(dtype= tf.float32, 
                            shape=(None, data_size[0], data_size[1], data_size[2]), name="inputs")
    
    # normalized inputs to the range of [-1, 1]
    normalized_inputs = normalize(inputs, name="input_normalization")
    
    # create a summary node for some of the inputs
    inputs_summary = tf.summary.image("Input", normalized_inputs[:8])
    # visualize only first 8 images from the batch

    
    # We feed the original inputs to the convNet as shown below:
    # encoder layers: 96 x 192
    conv1_1 = tf.layers.conv2d(inputs, 32, [7, 7], strides=(2, 2), 
                            padding="SAME", name="conv_layer1_1")
    
<<<<<<< HEAD
    bn1_1 = tf.contrib.layers.batch_norm(conv1_1)
    
    # record histogram summary:
    bn1_1_summary = tf.summary.histogram("bn1_1_summary", bn1_1)
=======
    bn1_1 = tf.layers.batch_normalization(conv1_1, name="batch_normalization1_1")
>>>>>>> 78a99623fce7eeaa58f18ba307452e3cae9714a7
    
    relu1_1 = tf.nn.relu(bn1_1, name="relu1_1")
    
    # 48 x 96
    conv1_2 = tf.layers.conv2d(relu1_1, 32, [5, 5], strides=(2, 2),
                            padding="SAME", name="conv_layer1_2")
    
<<<<<<< HEAD
    bn1_2 = tf.contrib.layers.batch_norm(conv1_2)
=======
    bn1_2 = tf.layers.batch_normalization(conv1_2, name="batch_normalization1_2")
>>>>>>> 78a99623fce7eeaa58f18ba307452e3cae9714a7
    
    relu1_2 = tf.nn.relu(bn1_2, name="relu1_2")
    
    # 24 x 48
    conv1_3 = tf.layers.conv2d(relu1_2, 32, [5, 5], strides=(2, 2),
                            padding="SAME", name="conv_layer1_3")
    
<<<<<<< HEAD
    bn1_3 = tf.contrib.layers.batch_norm(conv1_3)
=======
    bn1_3 = tf.layers.batch_normalization(conv1_3, name="batch_normalization1_3")
>>>>>>> 78a99623fce7eeaa58f18ba307452e3cae9714a7
    
    relu1_3 = tf.nn.relu(bn1_3, name="relu1_3")
    
    # 12 x 24
    conv1_4 = tf.layers.conv2d(relu1_3, 16, [3, 3], strides=(2, 2), 
                            padding="SAME", name="conv_layer1_4")

<<<<<<< HEAD
    bn1_4 = tf.contrib.layers.batch_norm(conv1_4)
    
    # record histogram summary:
    bn1_4_summary = tf.summary.histogram("bn1_4_summary", bn1_4)
=======
    bn1_4 = tf.layers.batch_normalization(conv1_4, name="batch_normalization1_4")
>>>>>>> 78a99623fce7eeaa58f18ba307452e3cae9714a7
    
    relu1_4 = tf.nn.relu(bn1_4, name="relu1_4")
    
    # 6 x 12
    
    
    # decoder layers:
    # 6 x 12
    deconv1_1 = tf.layers.conv2d_transpose(relu1_4, 32, [4, 4], strides=(2, 2), 
                                           padding="SAME", name="deconv_layer_1")
    
<<<<<<< HEAD
    deBn1_1 = tf.contrib.layers.batch_norm(deconv1_1)
    # record summary to see the values
    deBn1_1_summary = tf.summary.histogram("deBn1_1_summary", deBn1_1)
=======
    deBn1_1 = tf.layers.batch_normalization(deconv1_1, name="de_batch_normalization1_1")
>>>>>>> 78a99623fce7eeaa58f18ba307452e3cae9714a7
    
    deRelu1_1 = tf.nn.relu(deBn1_1, name="de_relu1_1")
    
    # 12 x 24
    deconv1_2 = tf.layers.conv2d_transpose(deRelu1_1, 32, [4, 4], strides=(2, 2), 
                                           padding="SAME", name="deconv_layer_2")
    
<<<<<<< HEAD
    deBn1_2 = tf.contrib.layers.batch_norm(deconv1_2)
=======
    deBn1_2 = tf.layers.batch_normalization(deconv1_2, name="de_batch_normalization1_2")
>>>>>>> 78a99623fce7eeaa58f18ba307452e3cae9714a7
    
    deRelu1_2 = tf.nn.relu(deBn1_2, name="de_relu1_2")
    
    # 24 x 48
    deconv1_3 = tf.layers.conv2d_transpose(deRelu1_2, 32, [4, 4], strides=(2, 2), 
                                           padding="SAME", name="deconv_layer_3")
    
<<<<<<< HEAD
    deBn1_3 = tf.contrib.layers.batch_norm(deconv1_3)
=======
    deBn1_3 = tf.layers.batch_normalization(deconv1_3, name="de_batch_normalization1_3")
>>>>>>> 78a99623fce7eeaa58f18ba307452e3cae9714a7
    
    deRelu1_3 = tf.nn.relu(deBn1_3, name="de_relu1_3")
    
    # 48 x 96
    deconv1_4 = tf.layers.conv2d_transpose(deRelu1_3, 32, [4, 4], strides=(2, 2),
                                           padding="SAME", name="deconv_layer_4")
    
<<<<<<< HEAD
    deBn1_4 = tf.contrib.layers.batch_norm(deconv1_4)
    
    # record the histogram summary:
    deBn1_4_summary = tf.summary.histogram("deBn1_4_summary", deBn1_4)
=======
    deBn1_4 = tf.layers.batch_normalization(deconv1_4, name="de_batch_normalization1_4")
>>>>>>> 78a99623fce7eeaa58f18ba307452e3cae9714a7
    
    deRelu1_4 = tf.nn.relu(deBn1_4, name="de_relu1_4")

    # 96 x 192
    deconv1_5 = tf.layers.conv2d_transpose(deRelu1_4, 3, [3, 3], strides=(1, 1),
                                           padding="SAME", name="deconv_layer_5")
    
    # normalize the predictions i.e deconv1_5 as mentioned above. and then use it for calculating the loss
    normalized_outputs = normalize(deconv1_5, name="output_normalization")
    
    # summary for the output image.
    output_image_summary = tf.summary.image("Output", normalized_outputs[:8]) # record corresponding outputs 
    # for the images.
    
    output = relu1_4 # get a hook on to the latent representation of the encoder
    
    # also generate the summary of the latent representations.
    output_summary = tf.summary.histogram("Latent_Representation", output)

    y_pred = normalized_outputs # output of the decoder
    y_true = normalized_inputs # input at the beginning

    # define the loss for this model:
    # calculate the loss and optimize the network
    loss = tf.norm(y_pred - y_true, ord="euclidean", name="eucledian_loss") # claculate the euclidean loss.
    
    # add a summary op for loss.
    loss_summary = tf.summary.scalar("Loss", loss)

    # using Adam optimizer for optimization
    train_op = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999).minimize(loss, name="train_op")
    
    # single op to generate all the summary data
    all_summaries = tf.summary.merge_all()
