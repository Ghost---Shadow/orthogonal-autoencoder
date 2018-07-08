import tensorflow as tf
import numpy as np

class CnnOrthogonalAutoencoder:
    def __init__(self,sess,input_dim,latent_dim,classes,
                 logs_dir='./logs',load_vectors=False,denoising=True):
        # Placeholders
        with tf.name_scope("placeholders"):
            latent_size = latent_dim[0] * latent_dim[1]
            input_size = input_dim[0] * input_dim[1]
            
            self.X = tf.placeholder(tf.float32,[None,input_size],name='X')
            self.X_reshape = tf.reshape(self.X,(-1,input_dim[0],input_dim[1],1),name='X_reshape')
                        
            self.Z = tf.placeholder(tf.float32,[None,latent_size],name='Z')
            self.Z_reshape = tf.reshape(self.Z,(-1,latent_dim[0],latent_dim[1],1),name='Z_reshape')
            
            self.labels = tf.placeholder(tf.float32,[None,classes],name="labels")                    
            self.means = tf.placeholder(tf.float32,[classes,latent_size],name="means")
            self.missing_eyes = tf.placeholder(tf.float32,[None,classes,classes],name="missing_eyes")
            self.orthogonal_loss_weight = tf.placeholder(tf.float32,name="orthogonal_loss_weight")
            self.keep_prob = tf.placeholder(tf.float32,name="keep_prob")

        # Host variables
        try:
            if load_vectors:
                self.means_host = np.genfromtxt('./vectors.csv', delimiter=',')
            else:
                self.means_host = np.zeros((classes,latent_size))
        except:
            self.means_host = np.zeros((classes,latent_size))
            
        self.orthogonal_loss_weight_host = 0
        self.orthogonal_loss_weight_gain = 2e-2
        self.orthogonal_loss_weight_max = 2
        self.classes = classes
        self.logs_dir = logs_dir
        self.keep_prob_host = 1.0
        self.denoising = True
        self.noise_scale = .5

        # Precomputed eyes
        self.precomputed_eyes = []
        for i in range(self.classes):
            e = np.eye(self.classes)
            e[i,i] = 0
            self.precomputed_eyes.append(e)
        
        # Weights
        with tf.name_scope("weights_and_biases"):
            self.W1 = tf.Variable(tf.random_normal([3,3,1,10]))
            self.W2 = tf.Variable(tf.random_normal([3,3,10,1]))

        # Graph variables
        with tf.name_scope("graph_variables"):
            #with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            self.encode = self._encoder(self.X_reshape)
            self.encode_reshaped = tf.reshape(self.encode,(-1,latent_size))
            self.decode = self._decoder(self.Z_reshape)
            self.autoencode = self._decoder(self.encode)
            self.calculate_mean = tf.reduce_mean(self.encode,axis=0,name="calculate_mean")
            self.calculate_mean_reshaped = tf.squeeze(tf.reshape(self.calculate_mean,(1,latent_size)))
            
            orthogonality_matrix = tf.matmul(self.means,tf.transpose(self.means))                                             
            self.orthogonality_matrix_image = tf.cast(tf.reshape(orthogonality_matrix * 255,(1,10,10,1)),tf.uint8)
            tf.summary.image('orthogonality_matrix_image',self.orthogonality_matrix_image)
            
            orthogonality = tf.reduce_sum(tf.squared_difference(tf.eye(self.classes),orthogonality_matrix))
            tf.summary.scalar('orthogonality',orthogonality)

        # Loss
        self.loss = self._get_loss()

        # Optimizers
        #self.optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(self.loss)
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

        # Accuracy
        self.accuracy = self._get_accuracy()        
        
        init = tf.global_variables_initializer()
        sess.run(init)
        self.summary = tf.summary.merge_all()
        self.summaryWriter = tf.summary.FileWriter(self.logs_dir,sess.graph)

    def fit(self,sess,batch):
        fetches = [self.optimizer,self.orthogonal_loss,self.autoencode_loss]
        
        if self.denoising:
            x = batch[0] + np.random.normal(size=batch[0].shape,loc=0,scale=self.noise_scale)
        else:
            x = batch[0]
            
        feed_dict = {self.X:x,
                     self.means:self.means_host,
                     self.missing_eyes: self._calculate_missing_eyes(batch[1]),
                     self.keep_prob:self.keep_prob_host,
                     self.orthogonal_loss_weight:self.orthogonal_loss_weight_host}
        
        _,orthogonal_loss,autoencode_loss = sess.run(fetches,feed_dict=feed_dict)
       
        return orthogonal_loss,autoencode_loss

    def evaluate_accuracy(self,sess,batch):
        feed_dict = {self.X:batch[0],
                     self.means:self.means_host,
                     self.missing_eyes: self._calculate_missing_eyes(batch[1]),
                     self.keep_prob:1.0,
                     self.orthogonal_loss_weight:self.orthogonal_loss_weight_host,
                     self.labels:batch[1]}
        
        a,self.summary_output = sess.run([self.accuracy,self.summary],
                                         feed_dict=feed_dict)
        return a

    def update_means(self,sess,batch):
        new_mean_matrix = []

        # Calculate new mean vectors for each class
        for i in range(self.classes):
            indexes = np.all((batch[1]-np.eye(10)[i]) == 0,axis=1)
            mean_vector = sess.run([self.calculate_mean_reshaped],
                                   feed_dict={self.X:batch[0][indexes],
                                              self.keep_prob:1.0})[0]
            mean_vector = mean_vector / np.linalg.norm(mean_vector)
            new_mean_matrix.append(mean_vector)

        # Calculate change
        new_mean_matrix = np.array(new_mean_matrix)
        change = np.sum(np.absolute(new_mean_matrix - self.means_host))
        self.means_host = self.means_host * .5 + new_mean_matrix * .5
        for i in range(len(self.means_host)):
            self.means_host[i] /= np.linalg.norm(self.means_host[i])

        # Increase weight of orthogonal loss
        self.orthogonal_loss_weight_host += self.orthogonal_loss_weight_gain
        self.orthogonal_loss_weight_host = min(self.orthogonal_loss_weight_host,
                                               self.orthogonal_loss_weight_max)
        return change

    def save_summaries(self,sess,batch,epoch):
        if epoch and self.summary_output:
            self.summaryWriter.add_summary(self.summary_output,epoch)

    def close(self):
        np.savetxt("cnn_vectors.csv", self.means_host, delimiter=",")
        self.summaryWriter.close()    
        
    def _encoder(self,X):
        with tf.name_scope('encoder'):
            #with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            layer_1 = tf.nn.sigmoid(tf.nn.conv2d(X,self.W1,[1,3,3,1],padding='SAME'))
            layer_1 = tf.nn.dropout(layer_1,self.keep_prob)
            layer_2 = tf.nn.sigmoid(tf.nn.conv2d(layer_1,self.W2,[1,1,1,1],padding='SAME'))
        tf.summary.histogram('latent',layer_2)
        return layer_2

    def _decoder(self,Z):
        with tf.name_scope('decoder'):
            #with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            batch_size = tf.shape(Z)[0]
            layer_3 = tf.nn.sigmoid(tf.nn.conv2d_transpose(Z,self.W2,
                                           [batch_size,10,10,10],[1,1,1,1],
                                           padding='SAME'))
            layer_3 = tf.nn.dropout(layer_3,self.keep_prob)
            layer_4 = tf.nn.sigmoid(tf.nn.conv2d_transpose(layer_3,self.W1,
                                           [batch_size,28,28,1],[1,3,3,1],
                                           padding='SAME'))
        return layer_4

    def _get_loss(self):
        with tf.name_scope('loss_and_optimization'):
            # Reconstruction loss
            self.autoencode_loss = tf.reduce_mean(tf.squared_difference(self.X_reshape,self.autoencode))
            tf.summary.scalar('autoencode_loss',self.autoencode_loss)

            # Orthogonal loss
            eyes_x_means = tf.einsum('aij,jk->aik',self.missing_eyes,self.means,name="eyes_x_means")
            normalized_latent = tf.nn.l2_normalize(self.encode_reshaped,1)
            eyes_x_means_x_latent = tf.einsum('aij,aj->ai',
                                              eyes_x_means,
                                              normalized_latent,
                                              name="eyes_x_means_x_latent")
            isq2 = 1/np.sqrt(2)
            #eyes_x_means_x_latent = tf.pow(eyes_x_means_x_latent/isq2,3)            
            eyes_x_means_x_latent -= tf.clip_by_value(eyes_x_means_x_latent,-isq2,isq2)
            self.orthogonal_loss = tf.reduce_mean(eyes_x_means_x_latent,
                                                 name="orthogonal_loss")
            tf.summary.scalar('orthogonal_loss',self.orthogonal_loss)

            # Total losss
            total_loss = self.orthogonal_loss_weight * self.orthogonal_loss\
                         + self.autoencode_loss
            tf.summary.scalar('loss',total_loss)
            
            return total_loss

    def _get_accuracy(self):
        with tf.name_scope('accuracy'):
            y = tf.einsum('ij,aj->ai',self.means,self.encode_reshaped)
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(self.labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy',accuracy)
        return accuracy

    def _calculate_missing_eyes(self,y):
        eyes = []
        for v in y:
            eyes.append(self.precomputed_eyes[np.argmax(v)])
        return np.array(eyes)      
