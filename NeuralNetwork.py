import time
import numpy as np
import matplotlib.pyplot as plt
import math


class Module:
    def __init__(self, _type):
        self.type = _type

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Module should implement call function")

    def backward(self, dA):
        raise NotImplementedError("Module should implement backward function")


class BCELoss(Module):
    def __init__(self):
        super().__init__('BCELoss')

    def __call__(self, *args, **kwargs):
        YP = args[0]
        Y = args[1]
        m = Y.shape[1]
        cost = (1. / m) * (-np.dot(Y, np.log(YP).T) - np.dot(1 - Y, np.log(1 - YP).T))
        cost = np.squeeze(cost)
        return cost

    def backward(self, args):
        YP = args[0]
        Y = args[1]
        return - (np.divide(Y, YP) - np.divide(1 - Y, 1 - YP))


class NLLLoss(Module):
    def __init__(self):
        super().__init__('NLLLoss')
        self.cache = None

    def __call__(self, *args, **kwargs):
        YP = args[0]
        Y = args[1]
        indices = tuple(np.transpose([(y, col) for col, y in enumerate(Y[0, :])]))
        choose_matix = np.zeros(YP.shape)
        choose_matix[indices] = -1.
        # choose_matix = choose_matix + np.ones(YP.shape)*1e-12
        self.cache = choose_matix * (1 / Y.shape[1])
        cost = choose_matix * YP
        cost = (1. / Y.shape[1]) * np.sum(cost)
        return cost

    def backward(self, args):
        return self.cache


class SoftMax(Module):
    def __init__(self):
        super().__init__('SoftMax')
        self.cache = None

    def __call__(self, *args, **kwargs):
        Z = args[0]  # [feature_num, batch_size]
        exp_Z = np.exp(Z)
        softmax_result = exp_Z / np.sum(exp_Z, axis=0)
        self.cache = softmax_result
        return softmax_result

    def backward(self, dA):
        dA_dZ = self.cache * (1 - self.cache)
        return dA * dA_dZ


class Log(Module):
    def __init__(self):
        super().__init__('Log')
        self.cache = None

    def __call__(self, *args, **kwargs):
        Z = args[0]
        self.cache = Z
        return np.log(Z)

    def backward(self, dA):
        dA_dZ = 1 / self.cache
        return dA * dA_dZ


class CrossEntropyLoss(Module):
    def __init__(self):
        super().__init__('CrossEntropyLoss')
        self.softmax = SoftMax()
        self.log = Log()
        self.nllloss = NLLLoss()
        self.cache = {}

    def __call__(self, *args, **kwargs):
        YP = args[0]
        Y = args[1]
        self.cache['softmax_A'] = self.softmax(YP)
        self.cache['log_A'] = self.log(self.cache['softmax_A'])
        self.cache['final'] = self.nllloss(self.cache['log_A'], Y)
        return self.cache['final']

    def backward(self, args):
        dAL = self.nllloss.backward(args)
        dlog_dA = self.log.backward(dAL)
        dsoftmax_dA = self.softmax.backward(dlog_dA)
        return dsoftmax_dA


class ReLu(Module):
    def __init__(self):
        super().__init__('ReLu')
        self.cache = None

    def __call__(self, *args, **kwargs):
        Z = args[0]
        self.cache = Z
        return np.maximum(0, Z)

    def backward(self, dA):
        dZ = np.array(dA, copy=True)
        dZ[self.cache <= 0] = 0
        return dZ


class Sigmoid(Module):
    def __init__(self):
        super().__init__('Sigmoid')
        self.cache = None

    def __call__(self, *args, **kwargs):
        Z = args[0]
        self.cache = Z
        return 1 / (1 + np.exp(-Z))

    def backward(self, dA):
        s = 1 / (1 + np.exp(-self.cache))
        return dA * s * (1 - s)


class Dropout(Module):
    def __init__(self, rate, feature_num):
        super().__init__('Dropout')
        self.rate = rate
        self.feature_num = feature_num
        self.cache = None

    def __call__(self, *args, **kwargs):
        A = args[0]
        self.cache = (np.random.rand(A.shape[0], A.shape[1]) <= (1 - self.rate)).astype(int)
        return np.multiply(A, self.cache)

    def backward(self, dA):
        return dA


class Linear(Module):
    def __init__(self, in_feature, out_feature):
        super().__init__('Linear')
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.W = np.random.randn(out_feature, in_feature) / np.sqrt(in_feature)
        self.b = np.zeros((out_feature, 1))
        self.cache = None
        self.dW = None
        self.db = None

    def __call__(self, *args, **kwargs):
        A = args[0]  # input_data
        Z = self.W.dot(A) + self.b
        self.cache = (A, self.W, self.b)
        return Z

    def backward(self, dZ):
        A, W, b = self.cache
        m = A.shape[1]
        self.dW = 1. / m * np.dot(dZ, A.T)
        self.db = 1. / m * np.sum(dZ, axis=1, keepdims=True)
        dA = np.dot(W.T, dZ)
        return dA


class Sequential(Module):
    def __init__(self, *args):
        # check
        super().__init__('Sequential')
        for item in args:
            if not isinstance(item, Module):
                raise ValueError("Sequential params should be Module instance")
        self.modules = list(args)

    def __call__(self, *args, **kwargs):
        A = args[0]
        dropout = False
        if 'dropout' in kwargs:
            dropout = kwargs['dropout']
        for i in range(len(self.modules)):
            cur_module = self.modules[i]
            if not dropout and cur_module.type == 'Dropout':
                continue
            A_prev = A
            A = cur_module(A_prev)
        return A

    def backward(self, dAL):
        cur_dA = dAL
        for i in range(len(self.modules) - 1, -1, -1):
            cur_module = self.modules[i]
            cur_dA = cur_module.backward(cur_dA)

    def step(self, lr):
        for i in range(len(self.modules)):
            cur_module = self.modules[i]
            if cur_module.type == 'Linear':
                cur_module.W = cur_module.W - lr * cur_module.dW
                cur_module.b = cur_module.b - lr * cur_module.db

    def fit(self, X, y, epoch, batch_size, lr=1e-3, loss_function='BCE', print_loss=False):
        if loss_function not in ['BCE', 'CrossEntropyLoss']:
            raise ValueError("currently, only support BCELoss")
        if loss_function == 'BCE':
            Loss_module = BCELoss()
        elif loss_function == 'CrossEntropyLoss':
            Loss_module = CrossEntropyLoss()
        train_data_size = X.shape[1]
        batch_num = math.ceil(train_data_size / batch_size)
        loss_log = []
        for i in range(epoch):
            epoch_mean_loss = 0
            for batch_index in range(batch_num):
                batch_slice = slice(batch_index * batch_size, (batch_index + 1) * batch_size)
                # forward
                yp = self(X[:, batch_slice], dropout=True)
                # backward
                loss = Loss_module(yp, y[:, batch_slice])
                dAL = Loss_module.backward([yp, y[:, batch_slice]])
                self.backward(dAL)
                self.step(lr)
                epoch_mean_loss = (epoch_mean_loss * batch_index + loss) / (batch_index + 1)
            loss_log.append(epoch_mean_loss)
            if print_loss:
                print(f'epoch {i}:loss = {epoch_mean_loss}')
        return loss_log

    def predict(self, X):
        m = X.shape[1]
        p = np.zeros((1, m))
        yp = self(X)
        if yp.shape[0] == 1:
            for i in range(0, yp.shape[1]):
                if yp[0, i] > 0.5:
                    p[0, i] = 1
                else:
                    p[0, i] = 0
        else:
            p = np.argmax(yp, axis=0).reshape((1, m))
        return p


if __name__ == '__main__':
    import sys
    import pandas as pd

    print(sys.argv[1])
    train_data = pd.read_csv(sys.argv[1]).to_numpy().T
    train_label = pd.read_csv(sys.argv[2]).to_numpy().T
    test_data = pd.read_csv(sys.argv[3]).to_numpy().T
    # test_label = pd.read_csv('./data/xor_test_label.csv').to_numpy().T
    model = Sequential(Linear(2, 32), ReLu(),
                       Linear(32, 256), ReLu(),
                       Linear(256, 64), ReLu(),
                       Linear(64, 32), ReLu(),
                       Linear(32, 1), Sigmoid())
    model.fit(train_data, train_label, 300, 128, lr=5e-2, print_loss=True, loss_function='BCE')
    # model.predict(train_data, train_label)
    result = model.predict(test_data).astype(int).T.tolist()
    result = pd.DataFrame(result)
    result.to_csv('./test_predictions.csv', index=False)
