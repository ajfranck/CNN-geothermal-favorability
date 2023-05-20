from imports import *
from model import *
from dataloader import *


model = NiN(lr=0.01).to(device)
model.load_state_dict(torch.load('MODEL/model.pth'))
model.eval()

#test if model is correct by predicting
dataloader = val_dataloader

size = len(dataloader.dataset)
num_batches = len(dataloader)
saved_pred = np.zeros(size)
saved_labels = np.zeros(size)

with torch.no_grad():
    for X, y in dataloader:
        #save the labels
        X = X.to(device)
        pred = model(X)
        for i in range(len(y)): 
            saved_labels[i] = y[i]
            saved_pred[i] = pred[i].argmax(0)


labels = saved_labels
scores = saved_pred

import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix

idx2class = {0: '0-25', 1: '25-50', 2: '50-200', 3: '200+'}

confusion_matrix_df = pd.DataFrame(confusion_matrix(y_train, scores)).rename(columns=idx2class, index=idx2class)

plt.figure(figsize=(10,8))
sns.heatmap(confusion_matrix_df, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('confusion_matrix.png')
