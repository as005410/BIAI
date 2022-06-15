import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

(img_train, lab_train), (img_test, lab_test) = tf.keras.datasets.fashion_mnist.load_data()


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#one hot
def to_categorical(x, n_col=None):
  if not n_col:
    n_col = np.amax(x) + 1
  
  one_hot = np.zeros((x.shape[0], n_col))
  one_hot[np.arange(x.shape[0]), x] = 1
  return one_hot

def accuracy(y_true, y_pred):
  return np.sum(y_true == y_pred, axis = 0) / len(y_true)

def batch_loader(X, y = None, batch_size=64):
  n_samples = X.shape[0]
  for i in np.arange(0, n_samples, batch_size):
    begin, end = i, min(i + batch_size, n_samples)
    if y is not None:
      yield X[begin:end], y[begin: end]
    else:
      yield X[begin:end]




y_train, y_test = to_categorical(lab_train.astype("int")), to_categorical(lab_test.astype("int"))
X_train, X_test = img_train / 255.0, img_test / 255.0

X_train, X_test = X_train.reshape(-1, 28*28), X_test.reshape(-1, 28*28)
X_train.shape, X_test.shape

n_input_dim = 28*28 # 784
n_out = 10 # 10 classes

#clases for activation functions
class CrossEntropy():
  def __init__(self): pass

  def loss(self, y, p):
    p = np.clip(p, 1e-15, 1- 1e-15)
    return -y*np.log(p) - (1 - y) * np.log(1- p)
  
  def gradient(self, y, p):
    p = np.clip(p, 1e-15, 1- 1e-15)
    return -(y/p) + (1 - y) / (1 - p)

class LeakyReLU():
  def __init__(self, alpha = 0.2):
    self.alpha = alpha
  
  def __call__(self, x):
    return self.activation(x)
  
  def activation(self, x):
    return np.where(x >= 0, x, self.alpha * x)
  
  def gradient(self, x):
    return np.where(x >= 0, 1, self.alpha)

class Softmax():
  def __init__(self): pass
  
  def __call__(self, x):
    return self.activation(x)
  
  def activation(self, x):
    e_x = np.exp(x - np.max(x, axis = -1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims = True)
  
  def gradient(self, x):
    # Error was in our softmax
    p = self.activation(x)
    return p * (1 - p)

class Activation():
  def __init__(self, activation, name="activation"):
    self.activation = activation
    self.gradient = activation.gradient
    self.input = None
    self.output = None
    self.name = name
  
  def forward(self, x):
    self.input = x
    self.output = self.activation(x)
    return self.output
  
  def backward(self, output_error, lr = 0.01):
    return self.gradient(self.input) * output_error
  
  def __call__(self, x):
    return self.forward(x)

class Linear():
  def __init__(self, n_in, n_out, name="linear"):
    limit = 1 / np.sqrt(n_in)
    self.W = np.random.uniform(-limit, limit, (n_in, n_out))
    self.b = np.zeros((1, n_out)) # Biases
    self.input = None
    self.output = None
    self.name = name
  
  def forward(self, x):
    self.input = x
    self.output = np.dot(self.input, self.W) + self.b 
    return self.output
  
  def backward(self, output_error, lr = 0.01):
    input_error = np.dot(output_error, self.W.T)
    delta = np.dot(self.input.T, output_error) 
    self.W -= lr * delta
    self.b -= lr * np.mean(output_error)
    return input_error
  
  def __call__(self, x):
    return self.forward(x)

class Network():
  def __init__(self, input_dim, output_dim, lr=0.01):
    self.layers = [
                   Linear(input_dim, 256, name="input"),
                   Activation(LeakyReLU(), name="relu1"),
                   Linear(256, 128, name="input"),
                   Activation(LeakyReLU(), name="relu2"),
                   Linear(128, output_dim, name="output"),
                   Activation(Softmax(), name="softmax")
    ]
    self.lr = lr
  
  def forward(self, x):
    for layer in self.layers:
      x = layer(x)
    return x
  
  def backward(self, loss_grad):
    for layer in reversed(self.layers):
      loss_grad = layer.backward(loss_grad, self.lr)
    #iterating backwards 
  
  def __call__(self, x):
    return self.forward(x)

criterion = CrossEntropy()
model = Network(n_input_dim, n_out, lr=1e-3)

EPOCHS = 1

for epoch in range(EPOCHS):
  loss = []
  acc = []
  for x_batch, y_batch in batch_loader(X_train, y_train):
    out = model(x_batch) #forward pass
    loss.append(np.mean(criterion.loss(y_batch, out))) #loss
    acc.append(accuracy(np.argmax(y_batch, axis=1), np.argmax(out, axis=1))) #accuracy
    error = criterion.gradient(y_batch, out)
    model.backward(error) #backpropagation
  
  print(f"Epoch {epoch + 1}, Loss: {np.mean(loss)}, Acc: {np.mean(acc)}")

out = model(X_test) #test set
print(accuracy(np.argmax(y_test, axis=1), np.argmax(out, axis=1)))

#print results
predicted_val = np.argmax(out, axis=1)

plt.figure(figsize=(5,5))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(img_test[i], cmap=plt.cm.binary)
    plt.xlabel("Predicted:"+class_names[predicted_val[i]])
    plt.title("True:"+ class_names[lab_test[i]])
plt.show()



