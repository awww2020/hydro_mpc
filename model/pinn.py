import abc
import tensorflow as tf
from model.nn import NN

class PINN(NN, metaclass=abc.ABCMeta):
    """
    Class used to represent a Physics informed Neural Network, children of NN.
    abc.ABCMeta 是 Python 内建模块 abc（Abstract Base Classes）的一部分，用于创建抽象基类。
    """

    def __init__(self, layers, lb, ub):
        """
        Constructor.

        :param list layers: widths of the layers
        :param np.ndarray lb: lower bounds of the inputs of the training data
        :param np.ndarray ub: upper bounds of the inputs of the training data
        """

        super().__init__(layers, lb, ub)

        self.loss_object = self.loss


    """
    def loss(self, y, y_pred, x=None):

        w_data = 1
        w_phys = 1

        f_pred = self.f_model(x)
        L_data = tf.reduce_mean(tf.square(y - y_pred))
        L_phys = tf.reduce_mean(tf.square(f_pred))


        L = w_data * L_data + w_phys * L_phys
        L = L_data
        return L
    """
    def loss(self, y, y_pred):

        w_data = 1
        w_phys = 1

        # f_pred = self.f_model()

        L_data = tf.reduce_mean(tf.square(y - y_pred))
        # L_phys = tf.reduce_mean(tf.square(f_pred))

        # L = w_data * L_data + w_phys * L_phys
        L = L_data

        return L

    def f_model(self, x):
        """
        Declaration of the function for the implementation of the f_model for a specific differential equation.
        """
        # 这边是为了方便继承的子类使用
        pass

    def predict(self, x):
        """
        Calls the model prediction function and returns the prediction on an input tensor.

        :param tf.tensor x: input tensor
        :return: tf.tensor: output tensor
        """
        return self.model_z.predict(x), self.model_u.predict(x), self.f_model(x)
