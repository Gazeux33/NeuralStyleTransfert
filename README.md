# Neural Style Transfert(NST)

<br>

This project is an implementation of Neural Style Transfer using PyTorch. Neural Style Transfer is a deep learning technique that merges the style of one image with the content of another, creating visually striking results by blending artistic features with photographic details.

## What is NST ? 

<div align="center">
	<img src="https://github.com/Gazeux33/NeuralStyleTransfert/blob/main/assets/principle.png" width="700">
</div>

<div align="center">
	<img src="https://github.com/Gazeux33/NeuralStyleTransfert/blob/main/assets/iterations.png" width="700">
</div>

## Technical specifications

| Property       | Value         |
|----------------|---------------|
| Framework      | PyTorch       |
| Device         | MAC M2        |
| Optimizer      | LBFGS         |
| Time for 1 image  | ~20 min    |

## How does it work ? 

### Extract characteristics with VGG19
The VGG19 pre-trained model is a convolutional neural network with 19 layers, including 16 convolutional layers and 3 fully connected layers, featuring 3x3 convolutional filters and max pooling, totaling approximately 143.67 million parameters for image classification tasks.

<div align="center">
	<img src="https://github.com/Gazeux33/NeuralStyleTransfert/blob/main/assets/features.png" width="700">
</div>

we start by extracting the characteristics of content and style

<div align="center">
	<img src="https://github.com/Gazeux33/NeuralStyleTransfert/blob/main/assets/howdoesitwork.png" width="700">
</div>

---

### Optimization Loop
<div align="center">
	<img src="https://github.com/Gazeux33/NeuralStyleTransfert/blob/main/assets/loop.png" width="700">
</div>
The ititial image can be:

- `content` 
- `style`
- `random noise`

---

### Loss Function
$$
L_{\text{total}} = \alpha L_{\text{content}} + \beta L_{\text{style}} + \gamma L_{\text{TV}}
$$

- `L_Content` is the content loss.
- `L_Style` is the style loss.
- `L_TV` is the total variation loss.
- `alpha`, `beta`, and  `gamma` are hyperparamters for each loss.

---

### Style Loss

<div align="center">
	<img src="https://github.com/Gazeux33/NeuralStyleTransfert/blob/main/assets/styleloss.png" width="700">
</div>

---

### Content Loss

<div align="center">
	<img src="https://github.com/Gazeux33/NeuralStyleTransfert/blob/main/assets/contentloss.png" width="700">
</div>

















Paper:<a href="https://arxiv.org/pdf/1705.04058" target="_blank">Neural Style Transfer: A Review</a>
