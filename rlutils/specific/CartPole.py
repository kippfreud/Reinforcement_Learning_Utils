from PIL import Image
import numpy as np
import torchvision.transforms as T
import torch

# Compose allows common image transforms to be chained together.
resize = T.Compose([T.ToPILImage(), # Convert a tensor or an ndarray to PIL Image.
                    T.Grayscale(),
                    T.Resize(40, interpolation=Image.CUBIC), # Scale the PIL Image so the shorter side matches the given value.
                    T.ToTensor()]) # Convert back to a tensor.

def screen_processor(screen, env, device=None):
    if device is None:
        print("WARNING: Device not specified, defaulting to best available device.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.4) : int(screen_height*0.8)]
    view_width = int(screen_width * 0.25)
    # Use the provided state representation to get the cart position in pixels.
    scale = screen_width / (env.x_threshold * 2) # Divide by world width.
    cart_location = int(env.state[0] * scale + screen_width / 2.0) 
    # Slice the screen to centre it on the cart.
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off edges so the image is square..
    screen = screen[:, :, slice_range]
    # Convert to float, rescale, convert to torch tensor.
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (batch, channel, height, width).
    return resize(screen).unsqueeze(0).to(device)