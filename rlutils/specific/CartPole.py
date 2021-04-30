from PIL import Image
import numpy as np
import torchvision.transforms as T
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Compose allows common image transforms to be chained together.
resize = T.Compose([T.ToPILImage(), # Convert a tensor or an ndarray to PIL Image.
                    T.Grayscale(),
                    T.Resize(40, interpolation=Image.CUBIC), # Scale the PIL Image so the shorter side matches the given value.
                    T.ToTensor()]) # Convert back to a tensor.

def screen_processor(screen, env):
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.415) : int(screen_height*0.8)]
    view_width = int(screen_width * 0.1)
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
    return resize(screen).unsqueeze(0).to(DEVICE)

def reward_function(state, action):
    """Reward function for CartPole-v1."""
    theta_threshold_radians = 12 * 2 * np.pi / 360
    x_threshold = 2.4
    x, _, theta, _ = state
    done = bool(x < -x_threshold
                or x > x_threshold
                or theta < -theta_threshold_radians
                or theta > theta_threshold_radians)
    return 1 if not done else 0