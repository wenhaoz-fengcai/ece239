
'''
Usage:
load single file:
train_loader, test_loader,val_loader = loader()(path='./project_datasets/A01T_slice.mat',
                                                batch_size= 20,
                                                num_test = 60,
                                                num_validation = 50)

load all file:
train_loader, test_loader,val_loader = loader()(path='ALL',
                                                batch_size= 20,
                                                num_test = 60,
                                                num_validation = 50)

'''

from torch.utils.data import TensorDataset
import h5py
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit


class loader(object):
    
    def __init__(self):
        self.path = None
        self.test_loader = None
        self.train_loader = None
        self.val_loader = None

        self.batch_size = None
        self.test_loaders = None
    def __call__(self,path='./project_datasets/A01T_slice.mat',batch_size = 30, num_test =50, num_validation =38):
        self.path = path
        self.batch_size = batch_size

        # only return the one file on the given path
        X,y = self.getAllData() if path == 'ALL' else self.getData()

        # make the loader 
        self.getLoader(X,y,num_test,num_validation)
        return self.train_loader, self.test_loader,self.val_loader
        


    
    def getData(self):

        data = h5py.File(self.path, 'r')
        # pick only the first 22 
        X = np.copy(data['image'])[:,:22,:]
        y = np.copy(data['type'])[:,:]
        print('X_shape',X.shape)
        print('y_shape',y.shape)

        
        y = y[0,0:X.shape[0]:1]
        y = np.asarray(y, dtype=np.int32)
        return X,y

    # use all data set to train
    def getAllData(self):

        paths = ['./project_datasets/A01T_slice.mat','./project_datasets/A02T_slice.mat',
                './project_datasets/A03T_slice.mat','./project_datasets/A04T_slice.mat',
                './project_datasets/A05T_slice.mat','./project_datasets/A06T_slice.mat',
                './project_datasets/A07T_slice.mat','./project_datasets/A08T_slice.mat',
                './project_datasets/A09T_slice.mat']

        # initial the empty array
        X = np.array([])
        y = np.array([])

        for path in paths:

            data = h5py.File(path, 'r')
            # pick only the first 22 
            X_temp = np.copy(data['image'])[:,:22,:]
            y_temp = np.copy(data['type'])[:,:]

            y_temp = y_temp[0,0:X_temp.shape[0]:1]
            y_temp = np.asarray(y_temp, dtype=np.int32)

            # stack all the X and y
            X = np.concatenate((X,X_temp),axis = 0) if X.size else X_temp
            y = np.concatenate((y,y_temp)) if y.size else y_temp

        return X,y
    
    
    
    def getAllDataSubject(self,num_test,num_validation):
        self.test_loaders = []        
        paths = ['./project_datasets/A01T_slice.mat','./project_datasets/A02T_slice.mat',
                './project_datasets/A03T_slice.mat','./project_datasets/A04T_slice.mat',
                './project_datasets/A05T_slice.mat','./project_datasets/A06T_slice.mat',
                './project_datasets/A07T_slice.mat','./project_datasets/A08T_slice.mat',
                './project_datasets/A09T_slice.mat']        
        X_train = np.array([])
        y_train = np.array([])
        X_test = np.array([])
        y_test = np.array([])
        X_val = np.array([])
        y_val = np.array([])
        for path in paths:

            data = h5py.File(path, 'r')
            # pick only the first 22 
            X_temp = np.copy(data['image'])[:,:22,:]
            y_temp = np.copy(data['type'])[:,:]

            y_temp = y_temp[0,0:X_temp.shape[0]:1]
            y_temp = np.asarray(y_temp, dtype=np.int32)
            # remove nan in x & y
            del_arr = []
            for i, row in  enumerate(X_temp):
                if np.isnan(np.sum(row)):
                    del_arr.append(i)

            X_temp = np.delete(X_temp,del_arr,0)
            y_temp = np.delete(y_temp,del_arr,0)
            for i in del_arr:
                print('nan exists on row {},and be deleted'.format(i)) 

            # map y to range(4)
            y_temp -= np.amin(y_temp)
            num_training = y_temp.shape[0] - num_test - num_validation
            print(X_temp.shape)
            balance = False


            sss = StratifiedShuffleSplit(n_splits=1, test_size=num_test, random_state=0)
            for train_index, test_index in sss.split(X_temp, y_temp):
                X_train_temp, X_test_temp = X_temp[train_index], X_temp[test_index]
                y_train_temp, y_test_temp = y_temp[train_index], y_temp[test_index]

            X_tp = X_train_temp.copy()
            y_tp = y_train_temp.copy()

            sss = StratifiedShuffleSplit(n_splits=1, test_size=num_validation, random_state=0)
            for train_index, test_index in sss.split(X_tp, y_tp):
                X_train_temp, X_val_temp = X_tp[train_index], X_tp[test_index]
                y_train_temp, y_val_temp = y_tp[train_index], y_tp[test_index]

            




            # stack all the X and y
            X_train = np.concatenate((X_train,X_train_temp),axis = 0) if X_train.size else X_train_temp
            y_train = np.concatenate((y_train,y_train_temp)) if y_train.size else y_train_temp
            X_test = np.concatenate((X_test,X_test_temp),axis = 0) if X_test.size else X_test_temp
            y_test = np.concatenate((y_test,y_test_temp)) if y_test.size else y_test_temp
            X_val = np.concatenate((X_val,X_val_temp),axis = 0) if X_val.size else X_val_temp
            y_val = np.concatenate((y_val,y_val_temp)) if y_val.size else y_val_temp     

        print('Train data shape: ', X_train.shape)
        print('Train labels shape: ', y_train.shape)
        print('test data shape: ', X_test.shape)
        print('test labels shape: ', y_test.shape)
        print('Validation data shape: ', X_val.shape)
        print('Validation labels shape: ', y_val.shape)
        # Normailize the data set 
        X_train  = (X_train - np.mean(X_train, axis=0))/ np.std(X_train,axis = 0)
        X_test  = (X_test - np.mean(X_test, axis=0))/ np.std(X_test,axis = 0)
        X_val  = (X_val - np.mean(X_val, axis=0))/ np.std(X_val,axis = 0)

        data_tensor = torch.Tensor(X_train.reshape(y_train.shape[0],1, 22, 1000))
        target_tensor = torch.Tensor(y_train)
        
        dataset = TensorDataset(data_tensor, target_tensor)

        # train
        self.train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                                   shuffle=True, sampler=None,
                                                   batch_sampler=None,
                                                   num_workers=0,
                                                   pin_memory=False, drop_last=False,
                                                   timeout=0, worker_init_fn=None)

        data_tensor = torch.Tensor(X_test.reshape(num_test*9,1, 22, 1000))
        target_tensor = torch.Tensor(y_test)

        # test
        dataset = TensorDataset(data_tensor, target_tensor)
        self.test_loader = torch.utils.data.DataLoader(dataset, batch_size=num_test,
                                                   shuffle=True, sampler=None,
                                                   batch_sampler=None,
                                                   num_workers=0,
                                                   pin_memory=False, drop_last=False,
                                                   timeout=0, worker_init_fn=None)
        for i in range(9):
            data_tensor = torch.Tensor(X_test[num_test*i:num_test*(i + 1)].reshape(num_test,1, 22, 1000))
            target_tensor = torch.Tensor(y_test[num_test*i:num_test*(i + 1)])
            dataset = TensorDataset(data_tensor, target_tensor)
            self.test_loaders.append(torch.utils.data.DataLoader(dataset, batch_size=num_test,
                                                       shuffle=True, sampler=None,
                                                       batch_sampler=None,
                                                       num_workers=0,
                                                       pin_memory=False, drop_last=False,
                                                       timeout=0, worker_init_fn=None))        

        # validation
        data_tensor = torch.Tensor(X_val.reshape(num_validation*9,1, 22, 1000))
        target_tensor = torch.Tensor(y_val)
        
        dataset = TensorDataset(data_tensor, target_tensor)
        self.val_loader = torch.utils.data.DataLoader(dataset, batch_size=num_validation,
                                                   shuffle=True, sampler=None,
                                                   batch_sampler=None,
                                                   num_workers=0,
                                                   pin_memory=False, drop_last=False,
                                                   timeout=0, worker_init_fn=None)


    def getLoader(self,X,y,num_test,num_validation):

        # remove nan in x & y
        del_arr = []
        for i, row in  enumerate(X):
            if np.isnan(np.sum(row)):
                del_arr.append(i)

        X = np.delete(X,del_arr,0)
        y = np.delete(y,del_arr,0)
        for i in del_arr:
            print('nan exists on row {},and be deleted'.format(i)) 

        # map y to range(4)
        y-= np.amin(y)

        # training set num
        num_training = y.shape[0] - num_test - num_validation


        # get balanced train test val dataset


        sss = StratifiedShuffleSplit(n_splits=1, test_size=num_test, random_state=0)
        for train_index, test_index in sss.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

        X_temp = X_train.copy()
        y_temp = y_train.copy()

        sss = StratifiedShuffleSplit(n_splits=1, test_size=num_validation, random_state=0)
        for train_index, test_index in sss.split(X_temp, y_temp):
            X_train, X_val = X_temp[train_index], X_temp[test_index]
            y_train, y_val = y_temp[train_index], y_temp[test_index]


        # balance = False
        # while not balance:

        #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=num_test)
        #     unique, counts = np.unique(y_test, return_counts=True)

        #     print('test',(np.amax(counts) - np.amin(counts)))
        #     balance = ((np.amax(counts) - np.amin(counts))<3)

        # balance = False
        # while not balance:
        #     X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=num_validation)
        #     unique, counts = np.unique(y_val, return_counts=True)

        #     print('test',(np.amax(counts) - np.amin(counts)))
        #     balance = ((np.amax(counts) - np.amin(counts))<3)



        print('Train data shape: ', X_train.shape)
        print('Train labels shape: ', y_train.shape)
        print('test data shape: ', X_test.shape)
        print('test labels shape: ', y_test.shape)
        print('Validation data shape: ', X_val.shape)
        print('Validation labels shape: ', y_val.shape)
        
        # Normailize the data set 
        mean_img = np.mean(X, axis=0)
        X_train  = (X_train - np.mean(X_train, axis=0))/ np.std(X_train,axis = 0)
        X_test  = (X_test - np.mean(X_test, axis=0))/ np.std(X_test,axis = 0)
        X_val  = (X_val - np.mean(X_val, axis=0))/ np.std(X_val,axis = 0)

        
        

        data_tensor = torch.Tensor(X_train.reshape(num_training,1, 22, 1000))
        target_tensor = torch.Tensor(y_train)
        
        dataset = TensorDataset(data_tensor, target_tensor)

        # train
        self.train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                                   shuffle=True, sampler=None,
                                                   batch_sampler=None,
                                                   num_workers=0,
                                                   pin_memory=False, drop_last=False,
                                                   timeout=0, worker_init_fn=None)

        data_tensor = torch.Tensor(X_test.reshape(num_test,1, 22, 1000))
        target_tensor = torch.Tensor(y_test)

        # test
        dataset = TensorDataset(data_tensor, target_tensor)
        self.test_loader = torch.utils.data.DataLoader(dataset, batch_size=num_test,
                                                   shuffle=True, sampler=None,
                                                   batch_sampler=None,
                                                   num_workers=0,
                                                   pin_memory=False, drop_last=False,
                                                   timeout=0, worker_init_fn=None)

        # validation
        data_tensor = torch.Tensor(X_val.reshape(num_validation,1, 22, 1000))
        target_tensor = torch.Tensor(y_val)
        
        dataset = TensorDataset(data_tensor, target_tensor)
        self.val_loader = torch.utils.data.DataLoader(dataset, batch_size=num_validation,
                                                   shuffle=True, sampler=None,
                                                   batch_sampler=None,
                                                   num_workers=0,
                                                   pin_memory=False, drop_last=False,
                                                   timeout=0, worker_init_fn=None)







