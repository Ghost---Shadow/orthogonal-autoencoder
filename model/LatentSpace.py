import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from model import OrthogonalAutoencoder

LATENT_SIZE = 100
CLASSES = 10
imgs = None
CHECKPOINT_DIR = './checkpoints/model'

def plotMultiple(imgs,dim,name):
    f,axarr = plt.subplots(dim[0],dim[1])
    f.canvas.set_window_title(name)
    for i in range(dim[0]):
        for j in range(dim[1]):
            axarr[i,j].imshow(imgs[i*dim[1]+j],interpolation='nearest',cmap='Greys')

with tf.Session() as sess:
    autoencoder = OrthogonalAutoencoder(sess,28*28,LATENT_SIZE,CLASSES)

    saver = tf.train.Saver()
    
    try:
        saver.restore(sess,CHECKPOINT_DIR)
    except Exception as e:
        print('Unable to restore model',e)

    mh = autoencoder.means_host
    
    imgs = sess.run(autoencoder.decode,{autoencoder.Z:mh})
    imgs = np.array(imgs).reshape(len(imgs),28,28) * 2.0
    plotMultiple(imgs,(len(imgs)//5,5),'Enumeration')
    
    latent = [mh[6] * (t/50) + mh[8] * (1-t/50) for t in range(20)]
    imgs = sess.run(autoencoder.decode,{autoencoder.Z:latent})
    imgs = np.array(imgs).reshape(len(imgs),28,28)
    plotMultiple(imgs,(len(imgs)//5,5),'Interpolation')

    vecs = np.random.randn(10,len(mh[0]))
    vecs /= np.linalg.norm(vecs, axis=0)    
    latent = [(mh[6] + vecs[t])/2 for t in range(10)]
    imgs = sess.run(autoencoder.decode,{autoencoder.Z:latent})
    imgs = np.array(imgs).reshape(len(imgs),28,28)
    plotMultiple(imgs,(len(imgs)//5,5),'Peturbation')
    
    plt.show()

    autoencoder.close()
