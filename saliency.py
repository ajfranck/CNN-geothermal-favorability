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


def relu_hook_function(module, grad_in, grad_out):
    if isinstance(module, torch.nn.ReLU):
        return (torch.clamp(grad_in[0], min=0.),)

for i, module in enumerate(model.modules()):
    if isinstance(module, torch.nn.ReLU):
        print(model.named_modules())
        module.register_backward_hook(relu_hook_function)

# Get the index corresponding to the maximum score and the maximum score itself.
score_max_index = scores.argmax()
score_max = scores[0,score_max_index]

score_max.backward()

grads = image.grad
print(grads)
GLOBAL_IMAGE = GLOBAL_IMAGE.squeeze(0).squeeze(0)
grads = grads.squeeze(0).squeeze(0)

def plot_maps(img1, img2,vmin=0.3,vmax=0.7, mix_val=2):
    f = plt.figure(figsize=(45,15))
    plt.subplot(1,3,1)
    plt.imshow(img1,vmin=vmin, vmax=vmax, cmap="gray")
    plt.axis("off")
    plt.subplot(1,3,2)
    plt.imshow(img2, cmap = "gray")
    plt.axis("off")
    plt.subplot(1,3,3)
    plt.imshow(img1*mix_val+img2/mix_val, cmap = "gray" )
    plt.axis("off")
    plt.show()

plot_maps(torch.Tensor.cpu(grads), torch.Tensor.cpu(GLOBAL_IMAGE))




# saliency, _ = torch.max(image.grad.data.abs(),dim=1)

# plt.subplot(1,2,1)
# # code to plot the saliency map as a heatmap
# plt.imshow(torch.Tensor.cpu(saliency[0]))#, cmap=plt.cm.hot)
# plt.axis('off')
# plt.suptitle('Saliency map, incorrect prediction')
# plt.savefig('saliency.png')

# plt.subplot(1,2,2)
# plt.imshow(saved_img)
# plt.axis('off')
# #title for both
# plt.show()