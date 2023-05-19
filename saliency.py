from imports import *
# from training import INIT_LR
from model import *
from dataloader import *
from torchcam.methods import SmoothGradCAMpp
from torchvision.transforms.functional import to_pil_image


NUMBER = 1
INIT_LR = 1e-3

file = h5py.File("data/DEM_train.h5", "r+")
X_train = np.array(file["/images"])
file.close()

original_image = torch.tensor(X_train[NUMBER,:,:]).to(torch.float)
saved_img = original_image
original_image = original_image[None, None,:,:]

#reshape to 1,200,200
GLOBAL_IMAGE = original_image

model = NiN(lr=INIT_LR).to(device)
model.load_state_dict(torch.load('MODEL/model.pth'))
model.eval()

#test if model is correct by predicting
image = GLOBAL_IMAGE.to(device)
image.requires_grad_()

scores = model(image)

labels = y_train
labels = labels[NUMBER]
print("Predicted: ", scores)
print("Actual: ", labels)


# Get the index corresponding to the maximum score and the maximum score itself.
score_max_index = scores.argmax()
score_max = scores[0,score_max_index]

score_max.backward()

saliency, _ = torch.max(image.grad.data.abs(),dim=1)

# plt.subplot(1,2,1)
fig, ax = plt.subplots()
# code to plot the saliency map as a heatmap
ax.imshow(saved_img)
ax.imshow(torch.Tensor.cpu(saliency[0]), alpha=0.5)
# plt.axis('off')
# plt.suptitle('Saliency map, incorrect prediction')
# plt.savefig('saliency.png')

# # plt.subplot(1,2,2)
# plt.imshow(saved_img)
# plt.axis('off')
# #title for both
plt.show()