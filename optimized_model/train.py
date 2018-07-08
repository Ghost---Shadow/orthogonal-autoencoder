import tensorflow as tf
import numpy as np

from optimized_model import OptimizedModel

EPOCHS = 2000
MINIBATCH_SIZE = 20
CHECKPOINT_EVERY = 50
LATENT_SIZE = 100
CLASSES = 10

PREPROCESSED_DIR = './preprocess/'
CHECKPOINT_DIR = './checkpoints/model'
LOGDIR = './logs/dropout'

data = [None for _ in range(CLASSES)]

for i in range(CLASSES):
    data[i] = np.load(PREPROCESSED_DIR+'train_preprocessed'+str(i)+'.npy')

def getBatch(size):
    batch = []

    for _ in range(size):
        b = []
        for c in range(CLASSES):
            idx = np.random.randint(len(data[c]))
            b.append(data[c][idx])
        batch.append(b)

    return np.array(batch,dtype=np.float32)

print('Started')
with tf.Session() as sess:
    autoencoder = OptimizedModel(sess,28*28,LATENT_SIZE,CLASSES,LOGDIR)

    for epoch in range(EPOCHS+1):
        batch = getBatch(MINIBATCH_SIZE)
        loss = autoencoder.fit(sess,batch,0.5)

        if epoch % CHECKPOINT_EVERY == 0 and epoch != 0:
            batch = getBatch(MINIBATCH_SIZE)
            print('Epoch %d: Total Loss: %.3f Orth wt: %.2f'%(epoch,loss,autoencoder.orthogonal_loss_weight_host))
            autoencoder.save_summaries(sess,batch,epoch,0.5)

    autoencoder.close()
    
# Alert when done
print('\a')
