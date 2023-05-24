from imports import *
from model import *
from dataloader import *


model = NiN(lr=0.01).to(device)
model.load_state_dict(torch.load('MODEL/model.pth'))
model.eval()

# #load test data
# file = h5py.File("data/DEM_test_features.h5", "r+")
# X_test = np.array(file["/images"])
# X_test = X_test[:,None,:,:]
# file.close()

# class ImageDataset:
#     def __init__(self, images):
#         self.X = torch.tensor(np.float32(images))
#     def __len__(self):
#         return self.X.shape[0]
#     def __getitem__(self, index):
#         return self.X[index, :, :, :]

# dataset_test = ImageDataset(X_test)

# test_dataloader = torch.utils.data.DataLoader(
#     dataset = dataset_test,
#     batch_size = 1,
#     shuffle = True)

#test if model is correct by predicting
dataloader = test_dataloader

size = len(dataloader.dataset)
num_batches = len(dataloader)
saved_pred = np.zeros(size)

with torch.no_grad():
    i = 0
    for X, y in dataloader:
        #save the labels
        X = X.to(device)
        pred = model(X)
        #save index of max value of pred into saved_pred
        saved_pred[i] = pred.argmax(1)
        i = i + 1

predictions = saved_pred
print(predictions)

import csv
#write predictions to csv file as one column with no header
with open('predictions.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(predictions)

