import numpy as np
from gym.envs.classic_control import rendering
from matplotlib import animation
import matplotlib.pyplot as plt


viewer = rendering.SimpleImageViewer()


def repeat_upsample(rgb_array, k=1, l=1, err=[]):
    # repeat kinda crashes if k/l are zero
    if k <= 0 or l <= 0:
        if not err:
            print("Number of repeats must be larger than 0, k: {}, l: {}, returning default array!".format(k, l))
            err.append('logged')
        return rgb_array

    # repeat the pixels k times along the y axis and l times along the x axis
    # if the input image is of shape (m,n,3), the output image will be of shape (k*m, l*n, 3)

    return np.repeat(np.repeat(rgb_array, k, axis=0), l, axis=1)


def render(env, scale=3):
    rgb = env.env.render('rgb_array')
    upscaled = repeat_upsample(rgb, scale, scale)
    viewer.imshow(upscaled)
    return upscaled


def save_frames_as_gif(frames, filename="results", scale=72.0):
    # https://gist.github.com/botforge/64cbb71780e6208172bbf03cd9293553

    # Had to scale it because I didn't have enough free memory
    plt.figure(figsize=(frames[0].shape[1] / scale, frames[0].shape[0] / scale), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(f"test_gifs/{filename}.gif", writer='imagemagick', fps=60)

