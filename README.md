# Synthetic-Image-Generation-Preprocessing-and-Flux-Model-Forward-Pass

## Overview
This project demonstrates the process of generating synthetic images using Stable Diffusion, preprocessing these images, and performing a forward pass through a simple Flux neural network model.

## 1. Synthetic Image Generation

### Approach
- Used Stable Diffusion model through the Hugging Face `diffusers` library to generate synthetic images
- Created images based on the prompt: "A serene sunset by a beach side"
- Generated 3 unique images and saved them with descriptive filenames

### Implementation Details
- Utilized Python with the `diffusers` library in Google Colab
- Implemented image generation with appropriate parameters for quality and diversity
- Saved the generated images in PNG format

## 2. Image Preprocessing

### Approach
- Loaded the generated images using Python's imaging libraries
- Resized all images to 224×224 pixels for consistency
- Normalized pixel values to the range [0, 1]
- Converted the images to tensors and saved them as NumPy arrays

### Implementation Details
- Used libraries such as PIL and NumPy for image manipulation
- Applied consistent preprocessing steps to all generated images
- Saved preprocessed images as NumPy files for compatibility with the next stage

## 3. Flux Model: Forward Pass Demonstration

### Approach
- Created a simple CNN using Julia's Flux library
- Used a model architecture with two convolutional layers, max pooling, and dense layers
- Performed a forward pass with the preprocessed images
- Visualized first-layer filters to demonstrate the model's feature detection

### Implementation Details
- Developed Julia scripts to run in Jupyter Lab
- Model architecture:
  * First convolutional layer: 3×3 kernel, input channels to 16 filters, ReLU activation
  * Max pooling: 2×2
  * Second convolutional layer: 3×3 kernel, 16 to 32 filters, ReLU activation
  * Max pooling: 2×2
  * Flatten layer
  * First dense layer: 32 * (H/4) * (W/4) to 64 neurons, ReLU activation
  * Output layer: 64 to 10 neurons with softmax activation
- Saved the model in BSON format for potential future use
- Created visualizations of the first layer filters

## Challenges and Solutions
### Flux Model
- Struggled to intergate Julia code execution within the Python-based Google Colab environment so swichetd to Jupyter Lab for further execution
- Ensuring correct tensor dimensions when working with Flux's convolutional layers
- Adapting the model architecture to handle different input sizes dynamically

## Running the Project
The initial setup for image generation and preprocessing was implemented in Google Colab to leverage its GPU resources for faster Stable Diffusion execution. Due to difficulties integrating Julia code execution within the Python-based Colab environment, the workflow was split across platforms:

- Generated the synthetic images in Google Colab using Python and the Diffusers library
- Performed preprocessing steps in Colab and exported the resulting tensors as NumPy files
- Downloaded the preprocessed image files locally
- Completed the Flux model implementation and forward pass demonstration using Jupyter Lab with a Python kernal itself

This hybrid approach allowed for optimal use of resources while overcoming difficulty integrating Julia code in Colab

## Environment Setup
### Google Colab Environment (Image Generation & Preprocessing)

- Python 3.8+
- Required libraries:
   - diffusers==0.14.0, 
   - transformers==4.25.1
   - torch==1.13.1,
   - numpy==1.22.4,
   - Pillow==9.4.0
- GPU [T4] runtime enabled for accelerated image generation

### Local Jupyter Lab Environment (Flux Model)
- Julia 1.8+
- Required Julia packages:
   - Flux.jl
   - CUDA.jl (if GPU acceleration is available)
   - Images.jl
   - FileIO.jl
   - BSON.jl
   - Plots.jl
   - NNlib.jl
   - NPZ.jl (for loading NumPy files)
- Jupyter Lab with IJulia kernel installed

### Installation Instructions
To replicate this environment locally:
- Install Julia from the official website
- Add the IJulia package to enable Julia in Jupyter: using Pkg; Pkg.add("IJulia")
- Install required packages: using Pkg; Pkg.add.(["Flux", "CUDA", "Images", "FileIO", "BSON", "Plots", "NNlib", "NPZ"])
- Launch Jupyter Lab and run the codes in python kernel itself when creating a new notebook
