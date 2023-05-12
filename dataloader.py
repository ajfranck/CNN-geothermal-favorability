from imports import *

# define the train and val splits
TRAIN_SPLIT = 0.8
VAL_SPLIT = 1 - TRAIN_SPLIT


BATCH_SIZE = 10

# Open training data and load
file = h5py.File("data/DEM_train.h5", "r+")
X_train = np.array(file["/images"])
y_train = np.array(file["/meta"])
file.close()
X_train = X_train[:,None,:,:]
X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train)

# Open test data and load
file = h5py.File("data/DEM_test_features.h5", "r+")
X_test = np.array(file["/images"])
X_test = X_test[:,None,:,:]
X_test = torch.tensor(X_test)
file.close()

#classify the y values
for i in range(len(y_train)):
    if y_train[i] <= 25:
        y_train[i] = 0
    elif y_train[i] <= 50:
        y_train[i] = 1
    elif y_train[i] <= 200:
        y_train[i] = 2
    else:
        y_train[i] = 3

#apply transform to data
transform1 = transforms.Compose([
    #transforms.RandomRotation(90),
    transforms.RandomHorizontalFlip(0.95)
])
#scripted_transform = torch.jit.script(transform1)
X_train_random = transform1(X_train)


#concatenate new data and original
X_train = torch.cat((X_train, X_train_random))
y_train = torch.cat((y_train, y_train))


class ImageDataset:
    def __init__(self, images, labels):
        self.y = torch.tensor(np.float32(labels))
        self.X = torch.tensor(np.float32(images))
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, index):
        return self.X[index, :, :, :], self.y[index]

dataset = ImageDataset(X_train, y_train)

numTrainSamples = math.ceil(len(dataset) * TRAIN_SPLIT)
numValSamples = int(len(dataset) * VAL_SPLIT)

(train_set, val_set) = torch.utils.data.random_split(dataset,
    [numTrainSamples, numValSamples],
	generator=torch.Generator().manual_seed(42))


#create dataloaders
train_dataloader = torch.utils.data.DataLoader(
    dataset = train_set,
    batch_size = BATCH_SIZE,
    shuffle = True)

val_dataloader = torch.utils.data.DataLoader(
    dataset = val_set,
    batch_size = BATCH_SIZE,
    shuffle = True)
