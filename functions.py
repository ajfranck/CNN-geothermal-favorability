from dataloader import *
from imports import *
import cv2

theta_deg = 60
transform1 = transforms.Compose([
    transforms.RandomRotation(theta_deg,interpolation=transforms.InterpolationMode.NEAREST),
    transforms.RandomHorizontalFlip(0.5),
])

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for i, (X, y) in enumerate(dataloader):
        y = y.type(torch.int64)
        X = transform1(X.to(device))
        y = y.to(device)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y).to(torch.float64)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            loss, current = loss.item(), (i + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return(loss_fn(pred, y).item())
    

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            y = y.type(torch.int64)
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, 100*correct


def create_cam(model, input, target_class):
    # Run the forward pass
    output = model(input.to(device))

    # Clear the gradients in the model parameters
    model.zero_grad()

    # Compute the loss
    loss = output[0, target_class]

    # Run the backward pass
    loss.backward()

    # Get the gradients of the target class with respect to the feature maps
    gradients = model.grad.clone().detach()

    # Pool the gradients across the channels
    pooled_gradients = gradients.mean(dim=[0, 2, 3])

    # Get the activations of the feature maps
    activations = model.fmap.clone().detach()

    # Weight the activations by the pooled gradients
    weighted_activation_maps = activations * pooled_gradients[:, None, None]

    # Calculate the class activation map
    cam = weighted_activation_maps.sum(dim=1).squeeze().detach()

    # Normalize the CAM to [0, 1]
    cam -= cam.min()
    cam /= cam.max()

    return cam


def overlay_cam_on_image(cam, image, prediction_correctness, smooth=True, blur_kernel_size=(5, 5)):
    cam = cam.detach().cpu().numpy()  # Convert CAM to a NumPy array

    # Resize the CAM to the size of the input image
    cam = cv2.resize(cam, (image.shape[3], image.shape[2]))

    # Apply smoothing to the CAM using Gaussian blur
    if smooth:
        cam = cv2.GaussianBlur(cam, blur_kernel_size, 0)

    # Normalize the smoothed CAM to be between 0 and 1
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))

    # Convert the CAM to the range [0, 255]
    cam = np.uint8(255 * cam)

    # Apply a colormap to the CAM
    cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    cam = cam/255.0

    # Convert the input image from PyTorch tensor to NumPy array
    image = image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

    # Overlay the CAM on the original image
    overlay_img = cv2.addWeighted(image.astype(np.float32), 0.7, cam.astype(np.float32), 0.3, 0, dtype=cv2.CV_32F)

    # Add a border based on prediction correctness
    border_color = (0, 255, 0) if prediction_correctness else (255, 0, 0)
    border_size = 10
    overlay_img = cv2.copyMakeBorder(overlay_img, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=border_color)

    return overlay_img
