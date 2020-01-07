#required modules and APIs
import os
from glob import glob
import cv2
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib import rnn
from sklearn.model_selection import train_test_split


#prepare the data by resizing the image to 28*28
img_path = os.getcwd() + '/data/'
folders = os.listdir(img_path)

#images and labels
X, y = [], []

#get the data in the required format
for folder in folders:
    files = glob(img_path+folder+'/*')
    for f in files:
        img_read = cv2.imread(f,0)
        img_resize = cv2.resize(img_read, (28, 28))
        thresh, blackAndWhiteImage = cv2.threshold(img_resize, 127, 255, cv2.THRESH_BINARY)
        X.append(blackAndWhiteImage)
        y.append(folder)

#dataset length
print("Dataset length: ", len(X), len(y))

#test train split
train_X, test_X, train_Y, test_Y = train_test_split(np.array(X), np.array(y), test_size=0.20, shuffle=True)

#train test details 
print('Training data shape : ', train_X.shape, train_Y.shape)
print('Testing data shape : ', test_X.shape, test_Y.shape)

#prepare data to feed into the network
train_X = train_X.reshape(-1, 28,28, 1)
test_X = test_X.reshape(-1, 28,28, 1)

#convert into float and normalize
train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_X = train_X / 255
test_X = test_X / 255

#convert the labels into one-hot
train_Y_one_hot = keras.utils.to_categorical(train_Y)
test_Y_one_hot = keras.utils.to_categorical(test_Y)

#Split data into training and validation
train_X, valid_X, train_label, valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2)
print('Train data shape : ', train_X.shape, train_label.shape)
print('Valid data shape : ', valid_X.shape, valid_label.shape)


logs_path = "./logs/write_summaries"  # path to the folder that we want to save the logs for Tensorboard

# Data Dimension
num_input = 28          #image resize, i.e., 28*28
timesteps = 28          #timesteps
n_classes = 14          #no of unqiue labels

learning_rate = 0.001   #the optimization initial learning rate
epochs = 10             #total number of training epochs
batch_size = 50         #training batch size
display_freq = 20       #frequency of displaying the training results

num_hidden_units = 128  #no of hidden units of the RNN

#shuffle the data
def randomize(x, y):
    permutation = np.random.permutation(y.shape[0])
    shuffled_x = x[permutation, :]
    shuffled_y = y[permutation]
    return shuffled_x, shuffled_y

#get next batch data
def get_next_batch(x, y, start, end):
    x_batch = x[start:end]
    y_batch = y[start:end]
    return x_batch, y_batch

#weights
def weight_variable(shape):
    initial = tf.truncated_normal_initializer(stddev=0.01)
    return tf.get_variable('W', dtype=tf.float32, shape=shape, initializer=initial)

#bias
def bias_variable(shape):
    initial = tf.constant(0., shape=shape, dtype=tf.float32)
    return tf.get_variable('b', dtype=tf.float32, initializer=initial)

#model
def BiRNN(x, weights, biases, timesteps, num_hidden):
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, num_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, num_input)
    x = tf.unstack(x, timesteps, 1)
    forward = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    backward = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    outputs, _, _ = rnn.static_bidirectional_rnn(forward, backward, x, dtype=tf.float32)
    #outputs,_=rnn.static_rnn(forward,x,dtype="float32")
    
    #Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights) + biases

# Placeholders for inputs (x) and outputs(y)
x = tf.placeholder(tf.float32, shape=[None, timesteps, num_input], name='X')
y = tf.placeholder(tf.float32, shape=[None, n_classes], name='Y')

# create weight matrix initialized randomely from N~(0, 0.01)
W = weight_variable(shape=[2*num_hidden_units, n_classes])

# create bias vector initialized as zero
b = bias_variable(shape=[n_classes])

output_logits = BiRNN(x, W, b, timesteps, num_hidden_units)
y_pred = tf.nn.softmax(output_logits)

# Model predictions
cls_prediction = tf.argmax(output_logits, axis=1, name='predictions')

# Define the loss function, optimizer, and accuracy
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output_logits), name='loss')
tf.summary.scalar('loss', loss)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam-op').minimize(loss)
correct_prediction = tf.equal(tf.argmax(output_logits, 1), tf.argmax(y, 1), name='correct_pred')
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
tf.summary.scalar('accuracy', accuracy)

# Creating the op for initializing all variables
init = tf.global_variables_initializer()
merged = tf.summary.merge_all()

valid_loss_list = []
train_loss_list = []
epoch_list = []

sess = tf.InteractiveSession()
sess.run(init)
global_step = 0
train_writer = tf.summary.FileWriter(logs_path + "/train", sess.graph)
#test_writer = tf.summary.FileWriter(logs_path + '/test')


# Number of training iterations in each epoch
num_tr_iter = int(len(train_label) / batch_size)
for epoch in range(epochs):
    print('Training epoch: {}'.format(epoch + 1))
    train_X, train_label = randomize(train_X, train_label)
    for iteration in range(num_tr_iter):
        global_step += 1
        start = iteration * batch_size
        end = (iteration + 1) * batch_size
        x_batch, y_batch = get_next_batch(train_X, train_label, start, end)
        x_batch = x_batch.reshape((batch_size, timesteps, num_input))
        # Run optimization op (backprop)
        feed_dict_batch = {x: x_batch, y: y_batch}
        _, summary_tr = sess.run([optimizer, merged], feed_dict=feed_dict_batch)
        train_writer.add_summary(summary_tr, global_step)

        if iteration % display_freq == 0:
            # Calculate and display the batch loss and accuracy
            loss_batch, acc_batch = sess.run([loss, accuracy], feed_dict=feed_dict_batch)

            print("iter {0:3d}:\t Loss={1:.2f},\tTraining Accuracy={2:.01%}".format(iteration, loss_batch, acc_batch))
    train_loss_list.append(loss_batch)
    # Run validation after every epoch
    feed_dict_valid = {x: valid_X.reshape((-1, timesteps, num_input)), y: valid_label}
    loss_valid, acc_valid = sess.run([loss, accuracy], feed_dict=feed_dict_valid)
    valid_loss_list.append(loss_valid)
    epoch_list.append(epoch+1)
    print('---------------------------------------------------------')
    print("Epoch: {0}, validation loss: {1:.2f}, validation accuracy: {2:.01%}".
          format(epoch + 1, loss_valid, acc_valid))
    print('---------------------------------------------------------')


def plot_loss_graph(epochs,loss = []):
    plt.title("Loss Vs Epochs Graph")
    plt.xlabel('No. of Epochs')
    plt.ylabel('Loss')
    plt.plot(epochs, loss[0],color="red", label="Training Loss")
    plt.plot(epochs, loss[1],color="blue", label="Validation Loss")
    plt.legend()
    plt.show()

def plot_images(images, cls_true, cls_pred=None, title=None):
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        ax.imshow(np.squeeze(images[i]).reshape(28, 28), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            ax_title = "True: {0}".format(cls_true[i])
        else:
            ax_title = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        ax.set_title(ax_title)
        ax.set_xticks([])
        ax.set_yticks([])
    if title:
        plt.suptitle(title, size=20)
    plt.show(block=False)

def plot_results(images, cls_true, cls_pred, title=None):
    correct = np.equal(cls_pred, cls_true)
    correct_images = images[correct]

    cls_pred = cls_pred[correct]
    cls_true = cls_true[correct]

    plot_images(images=correct_images[0:9], cls_true=cls_true[0:9], cls_pred=cls_pred[0:9], title=title)

def plot_example_errors(images, cls_true, cls_pred, title=None):

    incorrect = np.logical_not(np.equal(cls_pred, cls_true))

    incorrect_images = images[incorrect]
    cls_pred = cls_pred[incorrect]
    cls_true = cls_true[incorrect]
    
    plot_images(images=incorrect_images[0:9], cls_true=cls_true[0:9], cls_pred=cls_pred[0:9],title=title)


feed_dict_test = {x: test_X.reshape((-1, timesteps, num_input)), y: test_Y_one_hot}
loss_test, acc_test = sess.run([loss, accuracy], feed_dict=feed_dict_test)
print('---------------------------------------------------------')
print("Test loss: {0:.2f}, test accuracy: {1:.01%}".format(loss_test, acc_test))
print('---------------------------------------------------------')

# Plot training and validation loss
plot_loss_graph(epoch_list,[train_loss_list,valid_loss_list])

# Plot some of the correct and misclassified examples
cls_pred = sess.run(cls_prediction, feed_dict=feed_dict_test)
cls_true = np.argmax(test_Y_one_hot, axis=1)
plot_results(test_X, cls_true, cls_pred, title='Classified Examples')
plot_example_errors(test_X, cls_true, cls_pred, title='Misclassified Examples')
plt.show()

sess.close()




