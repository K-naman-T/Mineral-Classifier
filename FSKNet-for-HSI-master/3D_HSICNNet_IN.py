# -*- coding: utf-8 -*-
import numpy as np
import scipy.io as sio
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam, SGD, Adadelta, RMSprop, Nadam
import keras.callbacks as kcallbacks
import time
import collections
from sklearn import metrics, preprocessing
from Utils import zeroPadding, modelStatsRecord, averageAccuracy, cnn_3D_IN, HSI_CNN_IN, deformable_se_IN, \
    deformable_se_res_IN, deformable_se_sep_IN,hybrid_in,densenet_IN,cnn3_3D_IN,ssrn_SS_IN
import os
import matplotlib.pyplot as plt
import keras
from Utils.adabound import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


/////////
def indexToAssignment(index_, Row, Col, pad_length):
    new_assign = {}
    for counter, value in enumerate(index_):
        assign_0 = value // Col + pad_length
        assign_1 = value % Col + pad_length
        new_assign[counter] = [assign_0, assign_1]
    return new_assign


def assignmentToIndex(assign_0, assign_1, Row, Col):
    new_index = assign_0 * Col + assign_1
    return new_index


def selectNeighboringPatch(matrix, pos_row, pos_col, ex_len):
    selected_rows = matrix[range(pos_row - ex_len, pos_row + ex_len + 1), :]
    selected_patch = selected_rows[:, range(pos_col - ex_len, pos_col + ex_len + 1)]
    return selected_patch


# divide dataset into train and test datasets
#X =[x1, x2, x3, . . . , xB]^T ∈ R^(B×(N×M)) , where B represent total number of spectral bands consisting of (N ×M) samples per band belonging to Y classes where
def sampling(proptionVal, groundTruth):
    labels_loc = {}
    train = {}
    test = {}
    m = max(groundTruth)
    print(m)
    # 16
    # There are 16 categories. The samples of each category must be scrambled first, and then distributed in proportion to obtain a dictionary. Because the above is an enumeration, the corresponding sample and label
    for i in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
        # print(indices)
        # number of samples in each class
        np.random.shuffle(indices)
        labels_loc[i] = indices
        nb_val = int(proptionVal * len(indices))
        train[i] = indices[:-nb_val]
        test[i] = indices[-nb_val:]

    # Save all training samples into the train set and all test samples into the test set.
    train_indices = []
    test_indices = []
    for i in range(m):
        train_indices += train[i]
        test_indices += test[i]
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    print(len(test_indices))
    # 8194
    print(len(train_indices))
    # 2055
    return train_indices, test_indices


# Write a LossHistory class to save loss and acc
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()


# Calling the model structure
def model_HSICNNet():
    model_dense = densenet_IN.ResnetBuilder.build_resnet_8((1, img_rows, img_cols, img_channels), nb_classes)

    # abound = AdaBound()
    RMS = RMSprop(lr=0.0003)
    # Let's train the model using RMSprop
    model_dense.compile(loss='categorical_crossentropy', optimizer=RMS, metrics=['accuracy'])

    return model_dense


# Downloading Dataset
mat_data = sio.loadmat('F:/transfer code/Tensorflow  Learning/SKNet/datasets/IN/Indian_pines_corrected.mat')
data_IN = mat_data['indian_pines_corrected']
# Labeling Dataset
mat_gt = sio.loadmat('F:/transfer code/Tensorflow  Learning/SKNet/datasets/IN/Indian_pines_gt.mat')
gt_IN = mat_gt['indian_pines_gt']
# print('data_IN:',data_IN)
print(data_IN.shape)
# (145,145,200)
print(gt_IN.shape)
# (145,145)

new_gt_IN = gt_IN

batch_size = 16

nb_classes = 16
nb_epoch = 200  # 400
img_rows, img_cols = 23, 23  # 27, 27
patience = 200

INPUT_DIMENSION_CONV = 200
INPUT_DIMENSION = 200

TOTAL_SIZE = 10249
VAL_SIZE = 1025

TRAIN_SIZE = 5128
TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE
VALIDATION_SPLIT = 0.5  # 20% for trainnig and 80% for validation and testing
# 0.9  1031
# 0.8  2055
# 0.7  3081
# 0.6  4106
# 0.5  5128
# 0.4  6153

img_channels = 200
PATCH_LENGTH = 11  # Patch_size (13*2+1)*(13*2+1)

print(data_IN.shape[:2])
# (145,145)
print(np.prod(data_IN.shape[:2]))
# 21025
print(data_IN.shape[2:])
# (200,)
print(np.prod(data_IN.shape[2:]))
# 200
print(np.prod(new_gt_IN.shape[:2]))
# 21025

# After reshaping the data, perform the scale operation
data = data_IN.reshape(np.prod(data_IN.shape[:2]), np.prod(data_IN.shape[2:]))
gt = new_gt_IN.reshape(np.prod(new_gt_IN.shape[:2]), )

# Standardization operation, that is, all data are normalized along the rows and columns between 0-1
data = preprocessing.scale(data)
print(data.shape)
# (21025, 200)

# Filling the data edges is similar to the previous mirroring operation.
data_ = data.reshape(data_IN.shape[0], data_IN.shape[1], data_IN.shape[2])
whole_data = data_
padded_data = zeroPadding.zeroPadding_3D(whole_data, PATCH_LENGTH)
print(padded_data.shape)
# (151, 151, 200)
# Because the sliding window of 7*7 is selected, 145*145,145/7 leaves 5, which means that there are 5 pixels that cannot be scanned. All the length and width are filled with 3 on each side, which is 6. In this case All pixels can be scanned to ensure that the size of the input and output is the same.
ITER = 1
CATEGORY = 16

train_data = np.zeros((TRAIN_SIZE, 2 * PATCH_LENGTH + 1, 2 * PATCH_LENGTH + 1, INPUT_DIMENSION_CONV))
print(train_data.shape)
test_data = np.zeros((TEST_SIZE, 2 * PATCH_LENGTH + 1, 2 * PATCH_LENGTH + 1, INPUT_DIMENSION_CONV))
print(test_data.shape)

# Evaluation Index
KAPPA_3D_HSICNNet = []
OA_3D_HSICNNet = []
AA_3D_HSICNNet = []
TRAINING_TIME_3D_HSICNNet = []
TESTING_TIME_3D_HSICNNet = []
ELEMENT_ACC_3D_HSICNNet = np.zeros((ITER, CATEGORY))

# seeds = [1220, 1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228, 1229]
seeds = [1334]

for index_iter in range(ITER):
    print("# %d Iteration" % (index_iter + 1))
    # Iteration

    # save the best validated model
    best_weights_HSICNNet_path = 'F:/transfer code/Tensorflow  Learning/SKNet/models-in-densenet-23-514/Indian_best_3D_HSICNNet_' + str(
        index_iter + 1) + '.hdf5'

    # Get test and training samples through the sampling function
    np.random.seed(seeds[index_iter])
    train_indices, test_indices = sampling(VALIDATION_SPLIT, gt)
    # train_indices 2055     test_indices 8094


    # gt itself is a label class. Take the corresponding label -1 from the label class and convert it into one-hot form.    y_train = gt[train_indices] - 1
    y_train = to_categorical(np.asarray(y_train))

    y_test = gt[test_indices] - 1
    y_test = to_categorical(np.asarray(y_test))

    train_assign = indexToAssignment(train_indices, whole_data.shape[0], whole_data.shape[1], PATCH_LENGTH)
    for i in range(len(train_assign)):
        train_data[i] = selectNeighboringPatch(padded_data, train_assign[i][0], train_assign[i][1], PATCH_LENGTH)

    test_assign = indexToAssignment(test_indices, whole_data.shape[0], whole_data.shape[1], PATCH_LENGTH)
    for i in range(len(test_assign)):
        test_data[i] = selectNeighboringPatch(padded_data, test_assign[i][0], test_assign[i][1], PATCH_LENGTH)

    x_train = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], INPUT_DIMENSION_CONV)
    x_test_all = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], INPUT_DIMENSION_CONV)

    x_val = x_test_all[-VAL_SIZE:]
    y_val = y_test[-VAL_SIZE:]

    x_test = x_test_all[:-VAL_SIZE]
    y_test = y_test[:-VAL_SIZE]

    model_HSICNNet = model_HSICNNet()

    # loss history
    history = LossHistory()

    # monitor: monitoring data interface, here is val_loss, patience is how many steps can be tolerated without improvement changes    
    earlyStopping6 = kcallbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=1, mode='auto')
    # The user will save the model at the end of each epoch. If save_best_only=True, then the last data of the latest verification error will be saved.
    saveBestModel6 = kcallbacks.ModelCheckpoint(best_weights_HSICNNet_path, monitor='val_loss', verbose=1,
                                                save_best_only=True,
                                                mode='auto')

    # training and validation
    tic6 = time.clock()
    print(x_train.shape, x_test.shape)
    # (2055,7,7,200)  (7169,7,7,200)
    history_3d_HSICNNet = model_HSICNNet.fit(
        x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], x_train.shape[3], 1), y_train,
        validation_data=(x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], x_val.shape[3], 1), y_val),
        batch_size=batch_size,
        nb_epoch=nb_epoch, shuffle=True, callbacks=[earlyStopping6, saveBestModel6, history])
    toc6 = time.clock()

    # test
    tic7 = time.clock()
    loss_and_metrics = model_HSICNNet.evaluate(
        x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3], 1), y_test,
        batch_size=batch_size)
    toc7 = time.clock()

    print('3D HSICNNet Time: ', toc6 - tic6)
    print('3D HSICNNet Test time:', toc7 - tic7)

    print('3D HSICNNet Test score:', loss_and_metrics[0])
    print('3D HSICNNet Test accuracy:', loss_and_metrics[1])

    print(history_3d_HSICNNet.history.keys())

    # prediction
    pred_test = model_HSICNNet.predict(
        x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3], 1)).argmax(axis=1)
    # Tracks the number of occurrences of a value
    collections.Counter(pred_test)

    gt_test = gt[test_indices] - 1
    overall_acc = metrics.accuracy_score(pred_test, gt_test[:-VAL_SIZE])
    confusion_matrix = metrics.confusion_matrix(pred_test, gt_test[:-VAL_SIZE])
    each_acc, average_acc = averageAccuracy.AA_andEachClassAccuracy(confusion_matrix)
    kappa = metrics.cohen_kappa_score(pred_test, gt_test[:-VAL_SIZE])
    KAPPA_3D_HSICNNet.append(kappa)
    OA_3D_HSICNNet.append(overall_acc)
    AA_3D_HSICNNet.append(average_acc)
    TRAINING_TIME_3D_HSICNNet.append(toc6 - tic6)
    TESTING_TIME_3D_HSICNNet.append(toc7 - tic7)
    ELEMENT_ACC_3D_HSICNNet[index_iter, :] = each_acc

   #accuracy loss
    history.loss_plot('epoch')

    print("3D HSICNNet finished.")
    print("# %d Iteration" % (index_iter + 1))

# model statistics record
modelStatsRecord.outputStats(KAPPA_3D_HSICNNet, OA_3D_HSICNNet, AA_3D_HSICNNet, ELEMENT_ACC_3D_HSICNNet,
                             TRAINING_TIME_3D_HSICNNet, TESTING_TIME_3D_HSICNNet,
                             history_3d_HSICNNet, loss_and_metrics, CATEGORY,
                             'F:/transfer code/Tensorflow  Learning/SKNet/records-in-densenet-23-514/IN_train_3D.txt',
                             'F:/transfer code/Tensorflow  Learning/SKNet/records-in-densenet-23-514/IN_train_3D_element.txt')
