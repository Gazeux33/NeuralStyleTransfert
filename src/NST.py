from torch import nn
import torch
import os
import numpy as np
from torch.autograd import Variable
from torchvision.transforms import transforms
from PIL import Image
import matplotlib.pyplot as plt

import config
import utils
from vgg19 import Vgg19

print(torch.__version__)
IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
IMAGENET_STD_NEUTRAL = [1, 1, 1]


def stylisize_img(content_name, style_name, saving_features=False, save_frequences=50):
    
    # load images
    content_img = utils.load_image(os.path.join(config.CONTENT_DIR, content_name), config.HEIGHT)
    style_img = utils.load_image(os.path.join(config.STYLE_DIR, style_name), config.HEIGHT)

    # get the current device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # init the VGG19 model
    model = Vgg19()

    # transform images for VGG19
    content_transformed = transform_image(content_img, device)
    style_transformed = transform_image(style_img, device)

    #create the result dir for this image
    dir_name = content_name[:content_name.index(".")] + "_" + style_name[:style_name.index(".")]
    dir_path = os.path.join(config.RESULT_DIR, dir_name)
    os.makedirs(dir_path, exist_ok=True)
    
    # set the init image who will be optimize to reduce loss with the optimizer
    init_img = content_transformed
    optimizing_img = Variable(init_img, requires_grad=True)
    optimizer = torch.optim.LBFGS([optimizing_img], max_iter=config.ITERATION, line_search_fn="strong_wolfe")

    # get features map of the content and style
    content_set_features_map = model(content_transformed)
    style_set_features_map = model(style_transformed)
    
    # save theses features
    if saving_features:
        save_features(content_set_features_map, os.path.join(dir_path, "features_content"))
        save_features(style_set_features_map, os.path.join(dir_path, "features_style"))

    # get the target content and style(layer 4 for content and other for style )
    target_content = content_set_features_map[model.content_layers]
    target_style = [gram_matrix(x) for i, x in enumerate(style_set_features_map) if i in model.style_layers]

    # save grams matrix
    if saving_features:
        save_grams(target_style, os.path.join(dir_path, "grams_matrix"))

    # init counter and history for plots
    counter = 0
    total_loss_history = []
    content_loss_history = []
    style_loss_history = []
    tv_loss_history = []

    #definition of the closure function yo optimize the image(function for the optimizing loop)    
    def optimize_step():
        nonlocal counter
        optimizer.zero_grad()
        total_loss, content_loss, style_loss, tv_loss = loss_fn(model, target_content, target_style, optimizing_img)
        total_loss_history.append(total_loss.item())
        content_loss_history.append(content_loss.item())
        style_loss_history.append(style_loss.item())
        tv_loss_history.append(tv_loss.item())
        total_loss.backward()
        print(
            f"iteration:{counter} total_loss:{total_loss.item()} content_loss:{content_loss.item()} style_loss:{style_loss.item()} tv_loss:{tv_loss.item()}")
        if counter % save_frequences == 0:
            save_img(optimizing_img, os.path.join(dir_path, "results"), counter)
        counter += 1
        return total_loss

    # lauch the optimizing loop
    optimizer.step(optimize_step)
    save_img(optimizing_img, os.path.join(dir_path, "results"), counter)

    # plot results at the end 
    plot_results(
        {"content_loss": content_loss_history, "style_loss": style_loss_history,
         "tv_loss": tv_loss_history}, os.path.join(config.RESULT_DIR,dir_name, "plot.png"))



def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def loss_fn(model, target_content, target_style, current_img):
    current_img_set_features_map = model(current_img)

    # Content Loss
    content_loss_fn = torch.nn.MSELoss(reduction="mean")
    content_loss = content_loss_fn(target_content, current_img_set_features_map[4])

    # Style Loss
    style_loss_fn = torch.nn.MSELoss(reduction="sum")
    style_loss = 0.0
    current_style_representation = [gram_matrix(x) for i, x in enumerate(current_img_set_features_map) if i != 4]
    for gram_target, gram_current in zip(target_style, current_style_representation):
        style_loss += style_loss_fn(gram_target, gram_current)
    style_loss /= len(current_style_representation)

    # total variation loss
    tv_loss = total_variation(current_img)

    # Total loss
    total_loss = (content_loss * config.CONTENT_WEIGHT) + (style_loss * config.STYLE_WEIGHT) + (
            tv_loss * config.TV_WEIGHT)
    return total_loss, content_loss, style_loss, tv_loss


def total_variation(y):
    return torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + \
        torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))


def transform_image(img, device: torch.device) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
        transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL)
    ])
    img_transformed = transform(img).to(device).unsqueeze(0)
    return img_transformed


def save_features(data, path, nb=10):
    os.makedirs(path, exist_ok=True)
    for cnt, features in enumerate(data):
        features = features.squeeze()
        for i in range(nb):
            img = features[i].detach().numpy()
            name = f"features_{cnt}_{i}.png"
            plt.imsave(os.path.join(path, name), img, cmap="gray")


def save_grams(data, path):
    os.makedirs(path, exist_ok=True)
    for cnt, gram in enumerate(data):
        gram = gram.squeeze()
        img = gram.detach().numpy()
        name = f"gram_{cnt}.png"
        plt.imsave(os.path.join(path, name), img, cmap="gray")


def save_img(data, path, cnt):
    os.makedirs(path, exist_ok=True)
    img = data.squeeze().to('cpu').detach().numpy()
    img = np.moveaxis(img, 0, 2)
    name = f"optimizing_{cnt}.png"
    dump_img = np.copy(img)
    dump_img += np.array(IMAGENET_MEAN_255).reshape((1, 1, 3))
    dump_img = np.clip(dump_img, 0, 255).astype('uint8')
    plt.imsave(os.path.join(path, name), dump_img)


def plot_results(data, path):
    for key in data:
        plt.plot(data[key], label=key)
    plt.legend()
    plt.grid()
    plt.savefig(path)


if __name__ == "__main__":
    stylisize_img(config.content_name, config.style_name)
