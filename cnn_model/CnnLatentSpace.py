import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import imageio

from cnn_model import CnnOrthogonalAutoencoder

LATENT_SIZE = 100
CLASSES = 10
imgs = None
CHECKPOINT_DIR = './cnn_checkpoints/model'

def plotMultiple(imgs,dim,name):
    f,axarr = plt.subplots(dim[0],dim[1])
    f.canvas.set_window_title(name)
    for i in range(dim[0]):
        for j in range(dim[1]):
            axarr[i,j].imshow(imgs[i*dim[1]+j],interpolation='nearest',cmap='Greys')

with tf.Session() as sess:
    autoencoder = CnnOrthogonalAutoencoder(sess,(28,28),(10,10),CLASSES,load_vectors=True)

    saver = tf.train.Saver()
    
    try:
        saver.restore(sess,CHECKPOINT_DIR)
    except Exception as e:
        print('Unable to restore model',e)

    mh = autoencoder.means_host
    
    imgs = sess.run(autoencoder.decode,{autoencoder.Z:mh,autoencoder.keep_prob:1.0})
    imgs = np.array(imgs).reshape(len(imgs),28,28)
    plotMultiple(imgs,(len(imgs)//5,5),'Enumeration')

    steps = 20
    latent = []
    for i in range(9):
        latent.extend([mh[i+1] * (t/steps) + mh[i] * (1-t/steps) for t in range(steps)])
    latent.extend([mh[0] * (t/steps) + mh[9] * (1-t/steps) for t in range(steps)])
    imgs = sess.run(autoencoder.decode,{autoencoder.Z:latent,
                                        autoencoder.keep_prob:1.0})
    imgs = np.array(imgs).reshape(len(imgs),28,28)
    imgs = imgs / imgs.max()
    #plotMultiple(imgs,(len(imgs)//5,5),'Interpolation')
    imageio.mimsave('./interp.gif', np.uint8(imgs*255))
    
    vecs = np.random.randn(10,len(mh[0]))
    vecs /= np.linalg.norm(vecs, axis=0)    
    latent = [(mh[6] + vecs[t])/2 for t in range(10)]
    imgs = sess.run(autoencoder.decode,{autoencoder.Z:latent,autoencoder.keep_prob:1.0})
    imgs = np.array(imgs).reshape(len(imgs),28,28)
    plotMultiple(imgs,(len(imgs)//5,5),'Peturbation')
    
    plt.show()

    autoencoder.close()
