from imports import *
# from training import INIT_LR
from model import *
from dataloader import *
from torchcam.methods import SmoothGradCAMpp
from torchvision.transforms.functional import to_pil_image
from functions import *


INIT_LR = 1e-3
# img_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
#                                     transforms.Resize((200,200)),
#                                     transforms.ToTensor()])
NUMBER = 0

file = h5py.File("data/DEM_train.h5", "r+")
X_train = np.array(file["/images"])
file.close()

original_image = torch.tensor(X_train[NUMBER,:,:]).to(torch.float)
saved_img = original_image

original_image = original_image[None, None,:,:]
print(original_image.shape)

#reshape to 1,200,200
GLOBAL_IMAGE = original_image

input_tensor = GLOBAL_IMAGE.to(device)

model = NiN(lr=INIT_LR).to(device)
model.load_state_dict(torch.load('MODEL/model.pth'))
model.eval()

#test if model is correct by predicting
out = model(GLOBAL_IMAGE.to(device))
out = out.argmax()

labels = y_train
labels = labels[NUMBER]
print("Predicted: ", out)
print("Actual: ", labels)


cam = create_cam(model, GLOBAL_IMAGE, labels.to(torch.int64))
overlay_img = overlay_cam_on_image(cam, GLOBAL_IMAGE, 0)

fig, ax = plt.subplots()
ax.imshow(GLOBAL_IMAGE)
ax.imshow(overlay_img, alpha=0.5)

# cam_extractor = SmoothGradCAMpp(model)

# out = model(input_tensor)
# activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)


# from torchcam.utils import overlay_mask

# result = overlay_mask(to_pil_image(original_image), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
# plt.imshow(result); plt.axis('off'); plt.tight_layout(); plt.show()
# plt.savefig('cam.png')