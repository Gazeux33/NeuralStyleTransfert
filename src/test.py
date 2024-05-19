from pathlib import Path
import matplotlib.pyplot as plt
import os
import torch
from torch.autograd import Variable
from datetime import datetime
import numpy as np

import utils
from vgg19 import Vgg19
import config





 # load and reshape the content image
content_img = utils.load_image(os.path.join(config.CONTENT_DIR,config.content_name),config.HEIGHT)
# load and reshape the style image
style_img = utils.load_image(os.path.join(config.STYLE_DIR,config.style_name),config.HEIGHT) 

model = Vgg19() 


# transform the content image into tensor and more 
content_transformed = utils.transform_image(content_img,device) 
# transform the style image into tensor and more
style_transformed = utils.transform_image(style_img,device) 

init_img = content_transformed # define the init image (it can be random noise, content or style)
optimizing_img = Variable(init_img,requires_grad=True) # define the variables who will be optimize to reduce loss (we optimize image's pixels)

content_set_features_map = model(content_transformed)# get the set of features map of content image 
style_set_features_map = model(style_transformed)# get the set of features map of style image 

optimizer = torch.optim.LBFGS([optimizing_img],max_iter=config.ITERATION,line_search_fn="strong_wolfe")


target_content = content_set_features_map[config.target_content_layer]
target_style = [utils.gram_matrix(x) for i,x in enumerate(style_set_features_map) if i in config.target_style_layer]



def loss_fn(model, target_content, target_style, current_img):
    current_img_set_features_map = model(current_img)

    # Content Loss
    content_loss_fn = torch.nn.MSELoss(reduction="mean")
    content_loss = content_loss_fn(target_content, current_img_set_features_map[4])

    # Style Loss
    style_loss_fn = torch.nn.MSELoss(reduction="sum")
    style_loss = 0.0
    current_style_representation = [utils.gram_matrix(x) for i, x in enumerate(current_img_set_features_map) if i != 4]
    for gram_target, gram_current in zip(target_style, current_style_representation):
        style_loss += style_loss_fn(gram_target, gram_current)
    style_loss /= len(current_style_representation)
    
    #total variation loss
    tv = utils.total_variation(optimizing_img)

    # Total loss
    total_loss = (content_loss * config.CONTENT_WEIGHT) + (style_loss*config.STYLE_WEIGHT) + (tv*config.TV_WEIGHT)
    return total_loss


counter = 0

def optimize_step():
    global counter
    optimizer.zero_grad()
    total_loss = loss_fn(model, target_content, target_style, optimizing_img)
    total_loss.backward()
    print(f"iteration:{counter} total_loss:{total_loss.item()}")
    counter+=1
    return total_loss

# Perform optimization
optimizer.step(optimize_step)


final_img = optimizing_img.squeeze().permute(1,2,0).to('cpu').detach().numpy()
final_img = utils.get_uint8_range(final_img)
final_img /=255

utils.display_image(final_img)

result_name = utils.get_name(config.content_name,config.style_name)
result_img_path = os.path.join(config.RESULT_DIR, f"{result_name}.png")
plt.imsave(result_img_path, final_img)