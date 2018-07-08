import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../MNIST/data", one_hot = True)

def save_data(raw_data_x,raw_data_y,file_name):
    raw_data_y = np.argmax(raw_data_y,axis=1)

    data = [[] for _ in range(10)]

    for i,c in enumerate(raw_data_y):
        data[c].append(list(raw_data_x[i]))

    lens = [len(data[c]) for c in range(10)]
    print(lens)

    for i in range(10):
        np.save('./preprocess/'+file_name+str(i),data[i])

raw_data_x,raw_data_y = mnist.train.next_batch(mnist.train.num_examples)
save_data(raw_data_x,raw_data_y,'train_preprocessed')

raw_data_x,raw_data_y = mnist.test.next_batch(mnist.test.num_examples)
save_data(raw_data_x,raw_data_y,'test_preprocessed')
