import tensorflow as tf
import numpy as np

class OrthogonalAutoencoder:
    def __init__(self,sess,input_size,latent_size,classes,learning_rate=1e-3):
        # Placeholders
        with tf.name_scope("placeholders"):
            self.X = tf.placeholder(tf.float32,[None,input_size],name="X")
            self.Z = tf.placeholder(tf.float32,[None,latent_size],name="Z")
            self.labels = tf.placeholder(tf.float32,[None,classes],name="labels")
                    
            self.means = tf.placeholder(tf.float32,[classes,latent_size],name="means")
            self.missing_eyes = tf.placeholder(tf.float32,[None,classes,classes],name="missing_eyes")
            self.orthogonal_loss_weight = tf.placeholder(tf.float32,name="orthogonal_loss_weight")

        # Host variables
        try:
            self.means_host = np.genfromtxt('./vectors.csv', delimiter=',')
        except:
            self.means_host = np.zeros((classes,latent_size))
        self.orthogonal_loss_weight_host = 0
        self.orthogonal_loss_weight_gain = 2e-2
        self.classes = classes

        # Precomputed eyes
        self.precomputed_eyes = []
        for i in range(self.classes):
            e = np.eye(self.classes)
            e[i,i] = 0
            self.precomputed_eyes.append(e)

        # Weights
        with tf.name_scope("weights_and_biases"):
            #sizes = [input_size,int(input_size*.75 + latent_size*.25), int(input_size*.25 + latent_size*.75),latent_size]
            sizes = [input_size,input_size//2,latent_size]
            print(sizes)
            self.W1 = tf.Variable(tf.random_normal([sizes[0],sizes[1]]),name='W1')
            self.W2 = tf.Variable(tf.random_normal([sizes[1],sizes[2]]),name='W2')
            #self.W3 = tf.Variable(tf.random_normal([sizes[2],sizes[3]]),name='W3')
        
            self.B1 = tf.Variable(tf.random_normal([sizes[1]]),name='B1')
            self.B2 = tf.Variable(tf.random_normal([sizes[2]]),name='B2')
            #self.B3 = tf.Variable(tf.random_normal([sizes[3]]),name='B3')
            self.B_out = tf.Variable(tf.random_normal([sizes[0]]),name='B_out')

        # Graph variables
        with tf.name_scope("graph_variables"):
            self.encode = self._encoder(self.X)
            self.decode = self._decoder(self.Z)
            self.autoencode = self._decoder(self.encode)
            self.calculate_mean = tf.reduce_mean(self.encode,axis=0,name="calculate_mean")
            self.orthogonality_matrix = tf.cast(tf.reshape(tf.matmul(self.means,tf.transpose(self.means)) * 255,(1,10,10,1)),tf.uint8)
            tf.summary.image('orthogonality_matrix',self.orthogonality_matrix)

        # Loss
        self.loss = self._get_loss()

        # Optimizers
        #self.optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(self.loss)
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        # Accuracy
        self.accuracy = self._get_accuracy()        
        
        init = tf.global_variables_initializer()
        sess.run(init)
        self.summary = tf.summary.merge_all()
        self.summaryWriter = tf.summary.FileWriter('./logs',sess.graph)    

    def fit(self,sess,batch):
        fetches = [self.optimizer,self.orthogonality_matrix,
                   self.orthogonal_loss,self.autoencode_loss,self.summary]
        
        feed_dict = {self.X:batch[0],
                     self.means:self.means_host,
                     self.missing_eyes: self._calculate_missing_eyes(batch[1]),
                     self.orthogonal_loss_weight:self.orthogonal_loss_weight_host}
        
        _,_,orthogonal_loss,autoencode_loss,self.summary_output = sess.run(fetches,feed_dict=feed_dict)
       
        return orthogonal_loss,autoencode_loss

    def evaluate_accuracy(self,sess,batch):           
        a = sess.run(self.accuracy,feed_dict={self.X:batch[0],
                                              self.labels:batch[1],
                                              self.means:self.means_host})
        return a
    
    def update_means(self,sess,batch):
        new_mean_matrix = []

        # Calculate new mean vectors for each class
        for i in range(self.classes):
            indexes = np.all((batch[1]-np.eye(10)[i]) == 0,axis=1)
            mean_vector = sess.run([self.calculate_mean],
                                   feed_dict={self.X:batch[0][indexes]})[0]
            mean_vector = mean_vector / np.linalg.norm(mean_vector)
            new_mean_matrix.append(mean_vector)

        # Calculate change
        new_mean_matrix = np.array(new_mean_matrix)
        change = np.sum(np.absolute(new_mean_matrix - self.means_host))
        self.means_host = new_mean_matrix

        # Increase weight of orthogonal loss
        self.orthogonal_loss_weight_host += self.orthogonal_loss_weight_gain
        self.orthogonal_loss_weight_host = min(self.orthogonal_loss_weight_host,1)
        return change

    def save_summaries(self,sess,batch,epoch):
        if epoch and self.summary_output:
            self.summaryWriter.add_summary(self.summary_output,epoch)

    def close(self):
        np.savetxt("vectors.csv", self.means_host, delimiter=",")
        self.summaryWriter.close()

    def _encoder(self,X):
        with tf.name_scope('encoder'):
            layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(X,self.W1),self.B1))
            layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1,self.W2),self.B2))
            #layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2,self.W3),self.B3))
        tf.summary.histogram('latent',layer_2)
        return layer_2

    def _decoder(self,Z):
        with tf.name_scope('decoder'):
            layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(Z,tf.transpose(self.W2)),self.B1))
            layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1,tf.transpose(self.W1)),self.B_out))
            #layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2,tf.transpose(self.W1)),self.B_out))
        return layer_2

    def _get_loss(self):
        with tf.name_scope('loss_and_optimization'):
            # Reconstruction loss
            self.autoencode_loss = tf.reduce_mean(tf.squared_difference(self.X,self.autoencode))
            tf.summary.scalar('autoencode_loss',self.autoencode_loss)

            # Orthogonal loss
            eyes_x_means = tf.einsum('aij,jk->aik',self.missing_eyes,self.means,name="eyes_x_means")
            normalized_latent = tf.nn.l2_normalize(self.encode,1)
            eyes_x_means_x_latent = tf.einsum('aij,aj->ai',eyes_x_means,normalized_latent,name="eyes_x_means_x_latent")
            eyes_x_means_x_latent = eyes_x_means_x_latent - (1/np.sqrt(2))
            eyes_x_means_x_latent = tf.maximum(eyes_x_means_x_latent,0)
            self.orthogonal_loss = tf.reduce_mean(eyes_x_means_x_latent,name="orthogonal_loss")
            tf.summary.scalar('orthogonal_loss',self.orthogonal_loss)

            # Total losss
            total_loss = self.orthogonal_loss_weight * self.orthogonal_loss\
                         + self.autoencode_loss
            tf.summary.scalar('loss',total_loss)
            
            return total_loss

    def _get_accuracy(self):
        with tf.name_scope('accuracy'):
            y = tf.einsum('ij,aj->ai',self.means,self.encode)
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(self.labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            #tf.summary.scalar('accuracy',self.accuracy)
        return accuracy
    
    def _calculate_missing_eyes(self,y):
        eyes = []
        for v in y:
            eyes.append(self.precomputed_eyes[np.argmax(v)])
        return np.array(eyes)
