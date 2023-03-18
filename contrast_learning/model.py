import tensorflow as tf


class UnitNormLayer(tf.keras.layers.Layer):
    '''Normalize vectors (euclidean norm) in batch to unit hypersphere.
    '''

    def __init__(self):
        super(UnitNormLayer, self).__init__()

    def call(self, input_tensor):
        norm = tf.norm(input_tensor, axis=1)
        return input_tensor / tf.reshape(norm, [-1, 1])


class DenseLeakyReluLayer(tf.keras.layers.Layer):
    '''A dense layer followed by a LeakyRelu layer
    '''

    def __init__(self, n, alpha=0.3):
        super(DenseLeakyReluLayer, self).__init__()
        self.dense = tf.keras.layers.Dense(n, activation=None)
        self.lrelu = tf.keras.layers.LeakyReLU(alpha=alpha)

    def call(self, input_tensor):
        x = self.dense(input_tensor)
        return self.lrelu(x)


class Encoder(tf.keras.Model):
    '''An encoder network, E(·), which maps an augmented image x to a representation vector, r = E(x) ∈ R^{DE}
    '''

    def __init__(self, normalize=True, activation='relu'):
        super(Encoder, self).__init__(name='')
        if activation == 'leaky_relu':
            self.hidden1 = DenseLeakyReluLayer(256)
            self.hidden2 = DenseLeakyReluLayer(256)
        else:
            self.hidden1 = tf.keras.layers.Dense(256, activation=activation)
            self.hidden2 = tf.keras.layers.Dense(256, activation=activation)

        self.normalize = normalize
        if self.normalize:
            self.norm = UnitNormLayer()

    def call(self, input_tensor, training=False):
        x = self.hidden1(input_tensor, training=training)
        x = self.hidden2(x, training=training)
        if self.normalize:
            x = self.norm(x)
        print("Decode : {}".format(x))
        return x


class Projector(tf.keras.Model):
    '''
    A projection network, P(·), which maps the normalized representation vector r into a vector z = P(r) ∈ R^{DP} 
    suitable for computation of the contrastive loss.
    '''

    def __init__(self, n, normalize=True, activation='relu'):
        super(Projector, self).__init__(name='')
        if activation == 'leaky_relu':
            self.dense = DenseLeakyReluLayer(256)
            self.dense2 = DenseLeakyReluLayer(256)
        else:
            self.dense = tf.keras.layers.Dense(256, activation=activation)
            self.dense2 = tf.keras.layers.Dense(256, activation=activation)

        self.normalize = normalize
        if self.normalize:
            self.norm = UnitNormLayer()

    def call(self, input_tensor, training=False):
        x = self.dense(input_tensor, training=training)
        x = self.dense2(x, training=training)
        if self.normalize:
            x = self.norm(x)
        print("Projector : {}".format(x))
        return x


class SoftmaxPred(tf.keras.Model):
    '''For stage 2, simply a softmax on top of the Encoder.
    '''

    def __init__(self, num_classes=8):
        super(SoftmaxPred, self).__init__(name='')
        self.dense = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, input_tensor, training=False):
        return self.dense(input_tensor, training=training)


class MLP(tf.keras.Model):
    """A simple baseline MLP with the same architecture to Encoder + Softmax/Regression output.
    """

    def __init__(self, num_classes=8, normalize=True, regress=False, activation='relu'):
        super(MLP, self).__init__(name='')
        if activation == 'leaky_relu':
            self.hidden1 = DenseLeakyReluLayer(256)
            self.hidden2 = DenseLeakyReluLayer(256)
        else:
            self.hidden1 = tf.keras.layers.Dense(256, activation=activation)
            self.hidden2 = tf.keras.layers.Dense(256, activation=activation)
        self.normalize = normalize
        if self.normalize:
            self.norm = UnitNormLayer()
        if not regress:
            self.output_layer = tf.keras.layers.Dense(
                num_classes, activation='softmax')
        else:
            self.output_layer = tf.keras.layers.Dense(1)

    def call(self, input_tensor, training=False):
        x = self.hidden1(input_tensor, training=training)
        x = self.hidden2(x, training=training)
        # x = input_tensor
        if self.normalize:
            x = self.norm(x)
        preds = self.output_layer(x, training=training)
        return preds

    def get_last_hidden(self, input_tensor, training=False):
        """Get the last hidden layer before prediction.
        """
        x = self.hidden1(input_tensor, training=training)
        x = self.hidden2(x, training=training)
        if self.normalize:
            x = self.norm(x)
        return x


class GRU(tf.keras.Model):
    """
    GRU with the same architecture to Encoder + Softmax/Regression output.
    """

    def __init__(self, num_classes=8, normalize=True, regress=False, input_shape=(10,1), activation='relu'):
        super(GRU, self).__init__(name='')
        self.hidden1 = tf.keras.layers.GRU(128, activation=activation, return_sequences=True, input_shape=input_shape)
        self.hidden2 = tf.keras.layers.GRU(128, activation=activation, return_sequences=True)
        self.hidden3 = tf.keras.layers.GRU(128, activation=activation)
        self.normalize = normalize
        if self.normalize:
            self.norm = UnitNormLayer()
        if not regress:
            self.output_layer = tf.keras.layers.Dense(
                num_classes, activation='softmax')
        else:
            self.output_layer = tf.keras.layers.Dense(1)

    def call(self, input_tensor, training=False):
        x = self.hidden1(input_tensor, training=training)
        x = self.hidden2(x, training=training)
        x = self.hidden3(x, training=training)
        # x = input_tensor
        if self.normalize:
            x = self.norm(x)
        preds = self.output_layer(x, training=training)
        return preds

    def get_last_hidden(self, input_tensor, training=False):
        '''Get the last hidden layer before prediction.
        '''
        x = self.hidden1(input_tensor, training=training)
        x = self.hidden2(x, training=training)
        x = self.hidden3(x, training=training)
        if self.normalize:
            x = self.norm(x)
        return x


class LSTM(tf.keras.Model):
    """
    LSTM with the same architecture to Encoder + Softmax/Regression output.
    activation = 'relu', 'tanh'
    """

    def __init__(self, num_classes=8, normalize=True, regress=False, input_shape=(10,1), activation='relu'):
        super(LSTM, self).__init__(name='')
        self.hidden1 = tf.keras.layers.LSTM(128, activation=activation, return_sequences=True, input_shape=input_shape)
        self.hidden2 = tf.keras.layers.LSTM(128, activation=activation, return_sequences=True)
        self.hidden3 = tf.keras.layers.LSTM(128, activation=activation)
        self.normalize = normalize
        if self.normalize:
            self.norm = UnitNormLayer()
        if not regress:
            self.output_layer = tf.keras.layers.Dense(
                num_classes, activation='softmax')
        else:
            self.output_layer = tf.keras.layers.Dense(1)

    def call(self, input_tensor, training=False):
        x = self.hidden1(input_tensor, training=training)
        x = self.hidden2(x, training=training)
        x = self.hidden3(x, training=training)
        if self.normalize:
            x = self.norm(x)
        preds = self.output_layer(x, training=training)
        return preds

    def get_last_hidden(self, input_tensor, training=False):
        '''Get the last hidden layer before prediction.
        '''
        x = self.hidden1(input_tensor, training=training)
        x = self.hidden2(x, training=training)
        x = self.hidden3(x, training=training)
        if self.normalize:
            x = self.norm(x)
        return x


class SimpleRNN(tf.keras.Model):
    """
    SimpleRNN with the same architecture to Encoder + Softmax/Regression output.
    activation = 'relu', 'tanh'
    """

    def __init__(self, num_classes=8, normalize=True, regress=False, input_shape=(10,1), activation='relu'):
        super(SimpleRNN, self).__init__(name='')
        self.hidden1 = tf.keras.layers.SimpleRNN(128, activation=activation, return_sequences=True, input_shape=input_shape)
        self.hidden2 = tf.keras.layers.SimpleRNN(128, activation=activation, return_sequences=True)
        self.hidden3 = tf.keras.layers.SimpleRNN(128, activation=activation)
        self.normalize = normalize
        if self.normalize:
            self.norm = UnitNormLayer()
        if not regress:
            self.output_layer = tf.keras.layers.Dense(
                num_classes, activation='softmax')
        else:
            self.output_layer = tf.keras.layers.Dense(1)

    def call(self, input_tensor, training=False):
        x = self.hidden1(input_tensor, training=training)
        x = self.hidden2(x, training=training)
        x = self.hidden3(x, training=training)
        if self.normalize:
            x = self.norm(x)
        preds = self.output_layer(x, training=training)
        return preds

    def get_last_hidden(self, input_tensor, training=False):
        '''Get the last hidden layer before prediction.
        '''
        x = self.hidden1(input_tensor, training=training)
        x = self.hidden2(x, training=training)
        x = self.hidden3(x, training=training)
        if self.normalize:
            x = self.norm(x)
        return x


class CNN(tf.keras.Model):
    """
    CNN with the same architecture to Encoder + Softmax/Regression output.
    activation = 'relu', 'tanh'
    """

    def __init__(self, num_classes=8, normalize=True, regress=False, input_shape=(256,1), activation='relu'):
        super(CNN, self).__init__(name='')
        self.conv1 = tf.keras.layers.Conv1D(100, 2, padding='valid', activation='relu', input_shape=input_shape)
        self.maxpool1 = tf.keras.layers.MaxPool1D(pool_size=2)
        self.conv2 = tf.keras.layers.Conv1D(100, 2, padding='valid', activation='relu')
        self.maxpool2 = tf.keras.layers.MaxPool1D(pool_size=2)
        self.dropout1 = tf.keras.layers.Dropout(0.25)
        self.flatten = tf.keras.layers.Flatten()
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        self.hidden1 = tf.keras.layers.Dense(128, activation=activation)
        self.normalize = normalize
        if self.normalize:
            self.norm = UnitNormLayer()
        if not regress:
            self.output_layer = tf.keras.layers.Dense(
                num_classes, activation='softmax')
        else:
            self.output_layer = tf.keras.layers.Dense(1)

    def call(self, input_tensor, training=False):
        x = self.conv1(input_tensor, training=training)
        x = self.maxpool1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.maxpool2(x, training=training)
        x = self.dropout1(x, training=training)
        x = self.flatten(x, training=training)
        x = self.dropout2(x, training=training)
        x = self.hidden1(x, training=training)
        if self.normalize:
            x = self.norm(x)
        preds = self.output_layer(x, training=training)
        return preds

    def get_last_hidden(self, input_tensor, training=False):
        '''Get the last hidden layer before prediction.
        '''
        x = self.conv1(input_tensor, training=training)
        x = self.maxpool1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.maxpool2(x, training=training)
        x = self.dropout1(x, training=training)
        x = self.flatten(x, training=training)
        x = self.dropout2(x, training=training)
        x = self.hidden1(x, training=training)
        if self.normalize:
            x = self.norm(x)
        return x
