import numpy as np
DEFAULT_MODEL_PATH = "/content/drive/My Drive/IFT 6135/Question_1/Model"

class NN:
    
    def __init__(self, weight_init = 'glorot', hidden_dims = (500, 500),
                 n_hidden = 2, seed = 0):
        dims = [784] + list(hidden_dims) + [10]
        
        self.weight_init = weight_init
      
        if weight_init == 'zero':
            self.initialize_weights_zero(n_hidden, dims)
        
        if weight_init == 'normal':
            self.initialize_weights_normal(n_hidden, dims, seed)
        
        if weight_init == 'glorot':
            self.initialize_weights_glorot(n_hidden, dims, seed)

        self.L = n_hidden
    
    def initialize_weights_zero(self, n_hidden, dims):    
        self.W = []
        self.b = []
        for i in range(n_hidden+1):        
            self.W.append(np.zeros([dims[i+1], dims[i]]))
            self.b.append(np.zeros([dims[i+1], 1]))
      
    def initialize_weights_normal(self, n_hidden, dims, seed = 0):
        np.random.seed(seed)
        self.W = []
        self.b = []
        for i in range(n_hidden+1):
            self.W.append(np.random.randn(dims[i+1], dims[i]))
            self.b.append(np.zeros([dims[i+1], 1]))     
          
    def initialize_weights_glorot(self, n_hidden, dims, seed = 0):
        np.random.seed(seed)
        self.W = []
        self.b = []
        for i in range(n_hidden+1):
            dl = np.sqrt(6/(dims[i]+dims[i+1]))
            self.W.append(np.random.uniform(-dl, dl,(dims[i+1], dims[i])))
            self.b.append(np.zeros([dims[i+1], 1]))
    
    def activation(self, x):
        if self.weight_init == 'normal':
            # use logistic
            return 1/(1+np.exp(-x))
        else:
            # use reLu
            return np.maximum(0, x)

    def activ_grad(self, x):
        if self.weight_init == 'normal':
            # use logistic
            return np.exp(-x)/(1+np.exp(-x))**2
        else:
            # use reLu
            return(x > 0).astype(np.float)
    
    def loss(self, yHat, y):
        i = np.nonzero(y)
        return -np.sum(np.log(yHat)[i])/y.shape[0]

    def loss_grad(self, yHat, y):
        return -y/yHat
    
    def softmax(self, x):
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps, axis = 1, keepdims = True)
    
    def softmax_grad(self, x):
        n = x.shape[0]
        p = x.shape[1]
        y = self.softmax(x)
        J = np.ndarray((n, p, p))
        for i in range(n):
            for j in range(p):
                for k in range(p):
                    if j == k:
                        J[i,j,k] = y[i,j] * (1 - y[i,k])
                    else:
                        J[i,j,k] = -y[i,j] * y[i,k]
        return J
    
    def update(self, dJdW, dJdb):
        self.W -= np.multiply(self.learning_rate, dJdW)
        self.b -= np.multiply(self.learning_rate, dJdb)
    
    def forward(self, x, y, W, b, L):
        a = []
        h = []

        x = x.T
  
        for i in range(L+1):
            if i == 0:
                ac = b[i] + np.matmul(W[i], x)
            else:
                ac = b[i] + np.matmul(W[i], h[i-1].T)
      
            a.append(ac.T)
    
            if (i != L):
                h.append(self.activation(ac.T))
        
        yHat = self.softmax(a[L])
        l = self.loss(yHat, y)
  
        return l, yHat, a, h

    def backward(self, x, y, yHat, a, h, W, b, L):
        n = x.shape[0]

        dJdW = [0] * (L+1)
        dJdb = [0] * (L+1)

        g = self.loss_grad(yHat, y).reshape(y.shape[0], 1, y.shape[1])

        for i in reversed(range(L+1)):
            if i == L:
                sg = self.softmax_grad(a[L])
                g = np.matmul(g, sg)
            else:
                ag = self.activ_grad(a[i]).reshape(a[i].shape[0], 1, a[i].shape[1])
                g = np.multiply(g, ag)

            if i != 0:
                hi = h[i-1]
            else:
                hi = x

            hi = hi.reshape(hi.shape[0], 1, hi.shape[1])

            dJdW[i] = np.sum(np.matmul(g.reshape(g.shape[0], g.shape[2], 1), hi), 
                             axis = 0)/n

            dJdb[i] = np.sum(g, axis = 0).T/n

            g = np.matmul(g, W[i])

        g = np.sum(g, axis = 0)/n

        return g, dJdW, dJdb
            
    def train(self, train_loader, num_epoch, learning_rate):
        self.learning_rate = learning_rate
        
        self.loss_per_epoch = []
        
        for i in range(num_epoch):
            print("Starting epoch ", i)
            loss_per_batch = []
            for i, data in enumerate(train_loader):            
                inputs, labels = data
                inputs = inputs.numpy().reshape(DEFAULT_BATCH_SIZE,784)
                labels = labels.numpy()
            
                y = np.zeros([DEFAULT_BATCH_SIZE, 10])
                for i in range(DEFAULT_BATCH_SIZE):
                    y[i, labels[i]] = 1

                l, yHat, a, h = self.forward(inputs, y, self.W, self.b, self.L)
                g, dJdW, dJdb = self.backward(inputs, y, yHat, a, h, self.W, self.b, self.L)
                self.update(dJdW, dJdb)
                
                loss_per_batch.append(l)
                
            self.loss_per_epoch.append(np.mean(loss_per_batch))
    
    def test_accuracy(self, test_loader):
        total = 0
        correct = 0
        for i, data in enumerate(test_loader):
            inputs, labels = data
            inputs = inputs.numpy().reshape(DEFAULT_BATCH_SIZE,784)
            labels = labels.numpy()
            l, yHat, a, h = self.forward(inputs, labels, self.W, self.b, self.L)
            preds = np.argmax(yHat, axis = 1)
    
            correct += sum(preds == labels)
            total += len(labels)
        
        return(correct/total)
        