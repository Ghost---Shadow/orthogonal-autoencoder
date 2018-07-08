import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os

mnist = input_data.read_data_sets("../../MNIST/data", one_hot = True)
CLASSES = 10

from cnn_model import CnnOrthogonalAutoencoder

EPOCHS = 10000
MINIBATCH_SIZE = 100
MEAN_VECTOR_BATCH_SIZE = mnist.train.num_examples
CHECKPOINT_EVERY = 100
LATENT_SIZE = 100

CHECKPOINT_DIR = './checkpoints/model'
LOAD_CHECKPOINT = False

with tf.Session() as sess:
    autoencoder = CnnOrthogonalAutoencoder(sess,
                                           input_dim=(28,28),
                                           latent_dim=(10,10),
                                           classes=10,
                                           logs_dir='./logs/cnn/0.5/soft_clip/interpolate/denoising_0.5')
    autoencoder.keep_prob_host = 0.5
    
    saver = tf.train.Saver()

    if os.path.exists(CHECKPOINT_DIR) and LOAD_CHECKPOINT:
        try:
            saver.restore(sess,CHECKPOINT_DIR)
        except:
            print('Unable to restore model')
        
    for epoch in range(EPOCHS+1):
        batch = mnist.train.next_batch(MINIBATCH_SIZE)
        o_loss,r_loss = autoencoder.fit(sess,batch)        
        
        if epoch % CHECKPOINT_EVERY == 0 and epoch != 0:
            batch = mnist.train.next_batch(MEAN_VECTOR_BATCH_SIZE)
            change = autoencoder.update_means(sess,batch)
            accuracy = autoencoder.evaluate_accuracy(sess,mnist.test.next_batch(mnist.test.num_examples))
            accuracy *= 100
            print('Epoch %d:\tOrth Loss: %.3f\tRecon Loss: %.3f\tChange: %.3f\tOrth Loss Wt: %.2f\tAccuracy: %.2f%%'% \
                  (epoch,o_loss,r_loss,change,autoencoder.orthogonal_loss_weight_host,accuracy))
            autoencoder.save_summaries(sess,batch,epoch)
            saver.save(sess,CHECKPOINT_DIR)
    
    autoencoder.close()

# Alert when done
print('\a')

