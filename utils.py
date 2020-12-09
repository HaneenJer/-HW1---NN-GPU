import numpy as np


def sigmoid(x):
    """
     Parameters
     ----------
     x : np.array input data

     Returns
     -------
     np.array
         sigmoid of the input xn

     """
    return (1 / (1+np.exp(np.negative(x))))
    raise NotImplementedError("To be implemented")


def sigmoid_prime(x):
    """
         Parameters
         ----------
         x : np.array input data

         Returns
         -------
         np.array
             derivative of sigmoid of the input x

    """
    return (sigmoid(x) * (1-sigmoid(x)))
    raise NotImplementedError("To be implemented")


def random_weights(sizes):
    """
         Parameters
         ----------
         sizes : list of sizes

         Returns
         -------
         list
             list of xavier initialized np arrays weight matrices

    """
    return [xavier_initialization(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
    raise NotImplementedError("To be implemented")


def zeros_weights(sizes):
    """
         Parameters
         ----------
         sizes : list of sizes

         Returns
         -------
         list
             list of zero np arrays weight matrices

    """
    return list(np.zeros((sizes[i], sizes[i + 1])) for i in range(len(sizes) - 1))
    raise NotImplementedError("To be implemented")


def zeros_biases(list):
    """
         Parameters
         ----------
         sizes : list of sizes

         Returns
         -------
         list
             list of zero np arrays bias matrices

    """
    return [np.zeros(list[i]) for i in range(len(list))]
    raise NotImplementedError("To be implemented")


def create_batches(data, labels, batch_size):
    """
         Parameters
         ----------
         data : np.array of input data
         labels : np.array of input labels
         batch_size : int size of batch

         Returns
         -------
         list
             list of tuples of (data batch of batch_size, labels batch of batch_size)

    """
    return list((data[i:i+batch_size],labels[i:i+batch_size]) for i in range(0,len(data),batch_size))
    raise NotImplementedError("To be implemented")


def add_elementwise(list1, list2):
    """
         Parameters
         ----------
         list1 : np.array of numbers
         list2 : np.array of numbers

         Returns
         -------
         list
             list of sum of each two elements by index
    """
    return list(i+j for i, j in zip(list1,list2))
    raise NotImplementedError("To be implemented")

def xavier_initialization(m, n):
    xavier = 1 / (m ** 0.5)
    return np.random.uniform(low=-xavier, high=xavier, size=(m, n))
