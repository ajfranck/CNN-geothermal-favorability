from imports import *
# from training import INIT_LR
from model import *
from dataloader import *
from torchcam.methods import SmoothGradCAMpp
from torchvision.transforms.functional import to_pil_image


INIT_LR = 1e-3
img_transform = transforms.Compose([transforms.Resize((200,200)),
                                    transforms.ToTensor()])


file = h5py.File("data/DEM_train.h5", "r+")
X_train = np.array(file["/images"])
file.close()

original_image = torch.tensor(X_train[0,:,:]).to(torch.float)
saved_img = original_image
original_image = original_image[None, None,:,:]
print(original_image.dtype)

#reshape to 1,200,200
GLOBAL_IMAGE = original_image

input_tensor = GLOBAL_IMAGE.to(device)

model = NiN(lr=INIT_LR).to(device)
model.load_state_dict(torch.load('MODEL/model.pth'))
model.eval()

cam_extractor = SmoothGradCAMpp(model)

#with SmoothGradCAMpp(model) as cam_extractor:
    # Preprocess data and feed it to the model
out = model(input_tensor)
    # Retrieve the CAM by passing the class index and the model output
activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)

#plt.imshow(torch.Tensor.cpu(activation_map[0].squeeze(0)).numpy()); plt.axis('off'); plt.tight_layout(); plt.show()

from torchcam.utils import overlay_mask

result = overlay_mask(to_pil_image(original_image), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
plt.imshow(result); plt.axis('off'); plt.tight_layout(); plt.show()
plt.savefig('cam.png')