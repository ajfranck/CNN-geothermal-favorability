from imports import *

# define the train and val splits
TRAIN_SPLIT = 0.8
VAL_SPLIT = 1 - TRAIN_SPLIT

BATCH_SIZE = 20

# Open training data and load
file = h5py.File("data/DEM_train.h5", "r+")
X_train = np.array(file["/images"])
y_train = np.array(file["/meta"])
file.close()

#normalize X_train
X_train = X_train/np.max(X_train)

train_idx = np.arange(0,222)
np.random.shuffle(train_idx)
valid_idx = train_idx[0:int(TRAIN_SPLIT*len(train_idx))]
train_idx = train_idx[int(TRAIN_SPLIT*len(train_idx)):len(train_idx)]

X_valid = X_train[valid_idx,:,:]
y_valid = y_train[valid_idx]
saved_x_valid = X_valid
saved_y_valid = y_valid

X_train = X_train[train_idx,:,:]
y_train = y_train[train_idx]

X_train = X_train[:,None,:,:]
X_valid = X_valid[:,None,:,:]

X_train = torch.tensor(X_train)#.to(torch.uint8)
y_train = torch.tensor(y_train)#.to(torch.uint8)
X_valid = torch.tensor(X_valid)#.to(torch.uint8)
y_valid = torch.tensor(y_valid)#.to(torch.uint8)

# Open test data and load
file = h5py.File("data/DEM_test_features.h5", "r+")
X_test = np.array(file["/images"])
X_test = X_test[:,None,:,:]
X_test = torch.tensor(X_test)
file.close()

#classify the y values
if(MODEL_TYPE == 'classification'):
    for i in range(len(y_train)):
        if y_train[i] <= 25:
            y_train[i] = 0
        elif y_train[i] <= 50:
            y_train[i] = 1
        elif y_train[i] <= 200:
            y_train[i] = 2
        else:
            y_train[i] = 3
    for i in range(len(y_valid)):
        if y_valid[i] <= 25:
            y_valid[i] = 0
        elif y_valid[i] <= 50:
            y_valid[i] = 1
        elif y_valid[i] <= 200:
            y_valid[i] = 2
        else:
            y_valid[i] = 3


class ImageDataset:
    def __init__(self, images, labels):
        self.y = torch.tensor(np.float32(labels))
        self.X = torch.tensor(np.float32(images))
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, index):
        return self.X[index, :, :, :], self.y[index]

dataset_val = ImageDataset(X_train, y_train)
dataset_train = ImageDataset(X_valid, y_valid)

#create dataloaders
train_dataloader = torch.utils.data.DataLoader(
    dataset = dataset_train,
    batch_size = BATCH_SIZE,
    shuffle = True)

val_dataloader = torch.utils.data.DataLoader(
    dataset = dataset_val,
    batch_size = BATCH_SIZE,
    shuffle = True)
