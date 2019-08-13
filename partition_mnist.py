from utils import mnist_reader
from utils.download import download
import random
import pickle
import numpy as np

##download(directory="mnist", url="http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", extract_gz=True)
##download(directory="mnist", url="http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", extract_gz=True)
##download(directory="mnist", url="http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", extract_gz=True)
##download(directory="mnist", url="http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", extract_gz=True)

##folds = 5
##
###Split mnist into 5 folds:
##mnist = items_train = mnist_reader.Reader('mnist', train=True, test=True).items
##class_bins = {}
##random.shuffle(mnist)
##
##for x in mnist:
##    print (type(x[0]))
##    xx
##    if x[0] not in class_bins:
##        class_bins[x[0]] = []
##    class_bins[x[0]].append(x)
##
##mnist_folds = [[] for _ in range(folds)]
##
##for _class, data in class_bins.items():
##    count = len(data)
##    print("Class %d count: %d" % (_class, count))
##
##    count_per_fold = count // folds
##
##    for i in range(folds):
##        mnist_folds[i] += data[i * count_per_fold: (i + 1) * count_per_fold]
##
##
##print("Folds sizes:")
##for i in range(len(mnist_folds)):
##    print(len(mnist_folds[i]))
##
##    output = open('data_fold_%d.pkl' % i, 'wb')
##    pickle.dump(mnist_folds[i], output)
##    output.close()




intr = 5

stInd=0

##fp = open('/media/joshua/Data/python_codes/fingerprinting/internship_experiments/AnomalyDetectionUsingAutoencoder-master/data/single/0db/AllDev.txt')
##lines = fp.readlines()
##RFTrace = np.array([[float(v) for v in line.split()] for line in lines])
##print (RFTrace.shape)
RFTrace = np.load('/media/joshua/Data/python_codes/fingerprinting/internship_experiments/USRP_ID/allData/allData_0DB.npy')

windowSize = [512]#,64,128,256]
for ix in windowSize:
    print ("/////////////////////////////newWindowSize/////////////////////////")
    print ("ix_windowSize = ", ix)
    
    #endInd = len(RFTrace)
    numberOfTX=6
    stC=stInd
    #enC=numberOfTX*(enInd)
    enC = len(RFTrace)
    columnone = RFTrace[:,0]  
    columntwo = RFTrace[:,1]
    print ("enC, columnone, columntwo= ", enC, columnone.shape, columntwo.shape)

    miniBatchWindow=ix
    numberofminibatch=int(enC/miniBatchWindow)
    aX = columnone.reshape(numberofminibatch,miniBatchWindow)
    bX = columntwo.reshape(numberofminibatch,miniBatchWindow)
    print ("numberofminibatch, aX, bX", numberofminibatch, aX.shape, bX.shape)

    X = np.array(np.concatenate((aX, bX), axis=1))
    #X = X.reshape((len(X), 16, 16, ix)).transpose(0, 2, 3, 1)#transpose(0, 1, 3, 2)#

    labeleSize=int(numberofminibatch/numberOfTX)
    label0 = [0]*labeleSize
    label1 = [1]*labeleSize
    label2 = [2]*labeleSize
    label3 = [3]*labeleSize
    label4 = [4]*labeleSize
    label5 = [5]*labeleSize
    print ("labeleSize", labeleSize)

    alabel = label0 + label1 + label2 + label3 + label4 + label5
    blabel = np.array([alabel])
    label = list(blabel.T)

    print (X.shape)
    print (type(alabel[0]))#.shape
    
    data = []
    for i in range(len(X)):#lbl,x in (label, X):
        data.append(tuple([alabel[i],X[i].reshape(32, 32)]))
        
    print (len(data[0][1]))
mnist = data
folds = 5
#Split mnist into 5 folds:
class_bins = {}

random.shuffle(mnist)

for x in mnist:
    #print (type(x[0]))
    if x[0] not in class_bins:
        class_bins[x[0]] = []
    class_bins[x[0]].append(x)

mnist_folds = [[] for _ in range(folds)]

for _class, data in class_bins.items():
    count = len(data)
    print("Class %d count: %d" % (_class, count))

    count_per_fold = count // folds

    for i in range(folds):
        mnist_folds[i] += data[i * count_per_fold: (i + 1) * count_per_fold]


print("Folds sizes:")
for i in range(len(mnist_folds)):

    output = open('data_fold_%d.pkl' % i, 'wb')
    pickle.dump(mnist_folds[i], output)
    output.close()


