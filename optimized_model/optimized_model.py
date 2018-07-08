import numpy as np
import tensorflow as tf

class OptimizedModel:
    def __init__(self,sess,input_size,latent_size,classes,logdir):
        # Host variables
        self.classes = classes
        self.input_size = input_size        
        self.orthogonal_loss_weight_host = 0.0
        self.orthogonal_loss_weight_gain = 2e-2
        sizes = [input_size,input_size//2,latent_size]

        # Placeholders
        self.X = tf.placeholder(tf.float32,[None,self.classes,self.input_size],name='X')        
        #self.Z = tf.placeholder(tf.float32,[None,latent_size],name="Z")        
        self.orthogonal_loss_weight = tf.placeholder(tf.float32,name='orthogonal_loss_weight')
        self.keep_prob = tf.placeholder(tf.float32,name='keep_prob')
        
        # Weights
        self.W1 = tf.Variable(tf.random_normal([1,1,sizes[0],sizes[1]]),name='W1')
        self.W1_tiled = tf.tile(self.W1,[tf.shape(self.X)[0],self.classes,1,1],
                                name='W1_tiled')
        
        self.W2 = tf.Variable(tf.random_normal([1,1,sizes[1],sizes[2]]),name='W2')
        self.W2_tiled = tf.tile(self.W2,[tf.shape(self.X)[0],self.classes,1,1],
                                name='W2_tiled')

        # Biases
        self.B1 = tf.Variable(tf.random_normal([sizes[1]]),name='B1')
        self.B2 = tf.Variable(tf.random_normal([sizes[2]]),name='B2')
        self.B_out = tf.Variable(tf.random_normal([sizes[0]]),name='B_out')

        # Graph Variables
        self.encode = self._encoder(self.X)
        self.autoencode = self._decoder(self.encode)
        self.loss = self._get_loss()
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

        # TF initializers
        init = tf.global_variables_initializer()
        sess.run(init)
        self.summary = tf.summary.merge_all()
        self.summaryWriter = tf.summary.FileWriter(logdir,sess.graph)

    def fit(self,sess,batch,keep_prob):
        fetches = [self.loss,self.optimizer]
        feed_dict = {self.X:batch,
                     self.keep_prob:keep_prob,
                     self.orthogonal_loss_weight:self.orthogonal_loss_weight_host}
        loss,_ = sess.run(fetches,feed_dict)
        return loss

    def save_summaries(self,sess,batch,epoch,keep_prob):
        fetches = [self.optimizer,self.summary]
        feed_dict = {self.X:batch,
                     self.keep_prob:keep_prob,
                     self.orthogonal_loss_weight:self.orthogonal_loss_weight_host}
        _,summary_output = sess.run(fetches,feed_dict)
        self.summaryWriter.add_summary(summary_output,epoch)

        # Increase weight of orthogonal loss
        self.orthogonal_loss_weight_host += self.orthogonal_loss_weight_gain
        self.orthogonal_loss_weight_host = min(self.orthogonal_loss_weight_host,1)

    def _encoder(self,X):
        with tf.name_scope('encoder'):
            X_expanded = tf.expand_dims(X,axis=2,name='X_expanded')            
            #X_expanded = tf.nn.dropout(X_expanded,self.keep_prob)
            
            layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(X_expanded,self.W1_tiled),self.B1),name='layer_1')
            layer_1 = tf.nn.dropout(layer_1,self.keep_prob)
            
            layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1,self.W2_tiled),self.B2),name='layer_2')
        return tf.squeeze(layer_2,axis=2)

    def _decoder(self,Z):
        with tf.name_scope('decoder'):
            Z_expanded = tf.expand_dims(Z,axis=2,name='Z_expanded')
            #Z_expanded = tf.nn.dropout(Z_expanded,self.keep_prob)
            
            W1_tiled_transposed = tf.transpose(self.W1_tiled,[0,1,3,2],name='W1_T_T')
            W2_tiled_transposed = tf.transpose(self.W2_tiled,[0,1,3,2],name='W2_T_T')
            
            layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(Z_expanded,W2_tiled_transposed),
                                           self.B1),name='layer_3')
            layer_3 = tf.nn.dropout(layer_3,self.keep_prob)
            
            layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3,W1_tiled_transposed),
                                           self.B_out),name='layer_4')
        return tf.squeeze(layer_4,axis=2)

    def _get_loss(self):
        # Reconstruction loss
        with tf.name_scope('reconstruction_loss'):
            reconstruction_loss = tf.reduce_mean(tf.squared_difference(self.X,
                                                                       self.autoencode))
            tf.summary.scalar('reconstruction_loss',reconstruction_loss)
            
            reconstructed_img = tf.cast(tf.reshape(self.autoencode[5,5] * 255,(1,28,28,1)),
                                        tf.uint8)
            tf.summary.image('reconstructed_img',reconstructed_img)            

        # Orthogonal loss
        with tf.name_scope('orthogonal_loss'):
            normalized_latent = tf.nn.l2_normalize(self.encode,2,name='normalized_latent')
            orthogonality_matrices = tf.matmul(normalized_latent,
                                               normalized_latent,
                                               transpose_b=True,
                                               name='orthogonality_matrices')
            mean_difference = tf.reduce_mean(tf.squared_difference(orthogonality_matrices,
                                                                   tf.eye(self.classes)),
                                   axis=0)
            
            orthogonal_loss = tf.reduce_mean(mean_difference)
            tf.summary.scalar('orthogonal_loss',orthogonal_loss)
            
            mean_difference = tf.cast(tf.reshape(mean_difference * 255,(1,10,10,1)),tf.uint8,name='mean_difference')
            tf.summary.image('mean_difference',mean_difference)

        # Total loss
        loss = reconstruction_loss + self.orthogonal_loss_weight * orthogonal_loss
        tf.summary.scalar('total_loss',loss)
        return loss

    def close(self):
        self.summaryWriter.close()
    
            
            
        
            
