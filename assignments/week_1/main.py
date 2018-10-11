import tensorflow as tf
import pandas as pd
import numpy as np

def load_space_csv_data(file_name):
    df = pd.read_csv(file_name, delim_whitespace=True)
    cols = list(df.columns.values)
    return df, cols

N = 1;
m = 51;

df, cols = load_space_csv_data('poverty.txt')
PovPct = df['PovPct'].values
#A0 = np.transpose(np.array([df['PovPct'].values,np.ones(m)]))
A0 = np.transpose(np.array([df['PovPct'].values,df['ViolCrime'].values,np.ones(m)]))
b0 = np.transpose(df['Brth15to17'].values)

with tf.variable_scope('layer1', reuse=tf.AUTO_REUSE):
    A = tf.get_variable('A', initializer=A0)
    b = tf.get_variable('b', initializer=b0)

matrix = tf.matmul(tf.transpose(A),A)
inverse = tf.matrix_inverse(matrix)
regress = tf.matmul(
        tf.matmul(inverse, tf.transpose(A)),
        tf.expand_dims(b,axis = 1))

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    output = sess.run(regress)
    print(output)

import matplotlib.pyplot as plt

def plot_results(A0,b0,output,index):
    x = np.linspace(np.min(A0[:,index]),np.max(A0[:,index]),100)

    plt.plot(x,output[2]+output[index]*x)
    plt.plot(A0[:,index],b0,'o')
    plt.show()

for index in range(output.size-1):
    plot_results(A0,b0,output,index)
