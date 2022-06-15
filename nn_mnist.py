from main import NeuralNetwork
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from keras.datasets import fashion_mnist
from keras.utils import to_categorical

# load the MNIST dataset and apply min/max scaling to scale the
# pixel intensity values to the range [0, 1] (each image is
# represented by an 8 x 8 = 64-dim feature vector)
print("[INFO] loading MNIST (sample) dataset...")
# load train and test dataset

	# load dataset
(trainX, trainY), (testX, testY) = fashion_mnist.load_data()
	

	

# digits = datasets.load_digits()
# data = digits.data.astype("float")
# data = (data - data.min()) / (data.max() - data.min())
# print("[INFO] samples: {}, dim: {}".format(data.shape[0], data.shape[1]))


# scale pixels

	# convert from integers to floats
train_norm = trainX.astype('float32')
test_norm = testX.astype('float32')
	# normalize to range 0-1
train_norm = train_norm / 255.0
test_norm = test_norm / 255.0


# train the network
print("[INFO] training network...")
nn = NeuralNetwork([784, 64, 16, 10])
print("[INFO] {}".format(nn))
nn.train(trainX, trainY, epochs=1000)

# evaluate the network
print("[INFO] evaluating network...")
predictions = nn.predict(testX)
predictions = predictions.argmax(axis=1)
print(classification_report(testY.argmax(axis=1), predictions))