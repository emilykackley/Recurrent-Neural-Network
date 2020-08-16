# Recurrent-Neural-Network

## Logistic Regression
- This code uses a model from the MNIST(Mixed National Institute of Standard and Technology) database using one-hot encoding to make a prediction using Logistic Regression 
  ### Approach
  - Use the MNIST dataset from split it into data and labels
      * dataset = input_data.read_data_sets("/data/mnist", one_hot=True)
  - Define parameters and create tensor placeholders for x and y to accomodate batch size data points  
      * x = tf.placeholder(tf.float32, [batch, 784])
      * y = tf.placeholder(tf.float32, [batch, 10])
  - Create weights and bias
      * w = tf.Variable(tf.random_normal(shape = [784,10], stddev = 0.01), name = "weights")
      * b = tf.Variable(tf.zeros([1,10]), name = "bias")
  - Create a prediction model, loss function, and training optimizer
      * pred_model = tf.matmul(X, w) + b
      * loss = tf.reduce_mean(entropy)
      * opt = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss)
  - Create tensorflow session and contents written to a folder for 'logistic_reg' and begin training
  - Test training model to determine the accuracy of the model with the parameters established
