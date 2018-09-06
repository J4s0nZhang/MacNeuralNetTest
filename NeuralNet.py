from __future__ import print_function
import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon

from scipy.io import loadmat
from multiprocessing import cpu_count

from os import listdir
from os.path import isfile, join
mx.random.seed(3)

cpucount = cpu_count()
ctx = mx.cpu(cpucount)
#-----------------------------------------------------------------------------------------------------------------------
#loads datasets for the neural network
path_noA = '/media/neerajensritharan/C7-13-Nastaran/rjones/dataset/No Activity'
files_noA = [f for f in listdir(path_noA) if isfile(join(path_noA, f))]
noA_list = []

for x in files_noA:
    y = loadmat(path_noA+'/'+x)
    a = np.array(y['Data3D'])
    a /= np.max(a)
    noA_list.append(a)

label_noA = [1]*(len(files_noA))

path_A = '/media/neerajensritharan/C7-13-Nastaran/rjones/dataset/Activity'
files_A = [f for f in listdir(path_A) if isfile(join(path_A, f))]
A_list = []

for x in files_A:
    y = loadmat(path_A+'/'+x)
    a = np.array(y['Data3D'])
    a /= np.max(a)
    A_list.append(a)

label_A = [0]*(len(files_A))

#create the training dataset
train_dataset = mx.gluon.data.ArrayDataset(mx.nd.array(noA_list+A_list).expand_dims(1),mx.nd.array(label_noA+label_A).expand_dims(1))
#--------------------------------------------------------------------------------------------

#load the data for the validation dataset
path_noA_val = '/media/neerajensritharan/C7-13-Nastaran/rjones/testdataset/No Activity'
files_noA_val = [f for f in listdir(path_noA_val) if isfile(join(path_noA_val, f))]
noA_val_list = []

for x in files_noA_val:
    y = loadmat(path_noA_val+'/'+x)
    a = np.array(y['Data3D'])
    a /= np.max(a)
    noA_val_list.append(a)


label_noA_val = [1]*(len(files_noA_val))


path_A_val = '/media/neerajensritharan/C7-13-Nastaran/rjones/testdataset/Activity'
files_A_val = [f for f in listdir(path_A_val) if isfile(join(path_A_val, f))]
A_list_val = []

for x in files_A_val:
    y = loadmat(path_A_val+'/'+x)
    a = np.array(y['Data3D'])
    a /= np.max(a)
    A_list_val.append(a)

label_A_val = [0]*(len(files_A_val))

#create the validation dataset
val_dataset = mx.gluon.data.ArrayDataset(mx.nd.array(noA_val_list+A_list_val).expand_dims(1),mx.nd.array(label_noA_val+label_A_val).expand_dims(1))
#-----------------------------------------------------------------------------------------------------------
batch_size = 32
num_outputs = 1
num_fc = 1000

#dataloaders help pass data from the datasets to the neural network
#shuffle the training data and the validation data
train_data = mx.gluon.data.DataLoader(train_dataset, batch_size= batch_size,shuffle= True, num_workers = cpucount)
val_data = mx.gluon.data.DataLoader(val_dataset,batch_size= batch_size,shuffle= True, num_workers = cpucount)

#set up the structure of the neural network
net = gluon.nn.HybridSequential()
with net.name_scope():
    net.add(gluon.nn.Conv3D(channels=30, kernel_size=3,strides=2, activation='relu'))
    net.add(gluon.nn.MaxPool3D(pool_size=2, strides=2))
    net.add(gluon.nn.Conv3D(channels=50, kernel_size=1, activation='relu'))
    net.add(gluon.nn.MaxPool3D(pool_size=2, strides=2))
    net.add(gluon.nn.Flatten())

    net.add(gluon.nn.Dense(num_fc, activation="relu"))
    net.add(gluon.nn.Dense(num_fc, activation="relu"))
    net.add(gluon.nn.Dense(num_outputs, activation = 'sigmoid'))

net.hybridize()
net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx) #initialize random parameters

binary_cross_entropy = gluon.loss.SigmoidBinaryCrossEntropyLoss() #the loss function: binary cross entropy loss for binary classification
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': .001}) #The network optimizer and the learning rate

def evaluate_accuracy(data_iterator, net): #The accuracy evaluation function
    acc = mx.metric.Accuracy() #use mxnet's built in accuracy
    for i, (data, label) in enumerate(data_iterator): #get data from the dataloader
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx) #get the actual labels (ground truth)
        output = net(data) #get the ouput of the network (prediction)

        acc.update(preds=output, labels=label) #calculate accuracy
    return acc.get()[1]

epochs = 100 #epochs don't matter as well as network parameters with 95%+ will be saved
smoothing_constant = .01

#during training the network compares its output with the ground truth and changes its weight to minimize the loss function
# the weights are changed incrementally by
for e in range(epochs):
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        with autograd.record():
            output = net(data)
            loss = binary_cross_entropy(output, label)
        loss.backward()
        trainer.step(data.shape[0])


        curr_loss = nd.mean(loss).asscalar()
        moving_loss = (curr_loss if ((i == 0) and (e == 0))
                       else (1 - smoothing_constant) * moving_loss + smoothing_constant * curr_loss)

    val_accuracy = evaluate_accuracy(val_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)

    if val_accuracy > 0.95: #save all network parameters with over 95 validation accuracy
        filename = 'fusnet'
        net.export(filename, epoch = e)

    print("Epoch %s. Loss: %s, Train_acc %s, Validation_acc %s" % (e, moving_loss, train_accuracy, val_accuracy))
