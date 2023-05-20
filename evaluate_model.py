from imports import *
from model import *
from dataloader import *


model = NiN(lr=0.01).to(device)
model.load_state_dict(torch.load('MODEL/model.pth'))
model.eval()

#test if model is correct by predicting
scores = np.zeros(len(saved_x_valid))

for i in range(len(saved_x_valid)):
    scores[i] = model(saved_x_valid[i,:,:])

labels = saved_y_valid

#compare labels to scores using abs(labels - scores)
def accuracy(y_hat, y):
    return np.abs(y_hat - y).mean()

print("Accuracy: ", accuracy(scores, labels))