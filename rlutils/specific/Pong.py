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
    _, screen_height, screen_width = screen.shape
    # Crop score.
    # screen = screen[:, int(screen_height*0.17) : int(screen_height*0.92)]
    # Keep score.
    screen = screen[:, : int(screen_height*0.92)]
    # Convert to float, rescale, convert to torch tensor.
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (batch, channel, height, width).
    return resize(screen).unsqueeze(0).to(device)