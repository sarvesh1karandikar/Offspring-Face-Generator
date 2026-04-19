# Offspring Face Generator

**Predict what a child might look like given photos of both parents using deep learning.**

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.x-red?logo=keras)](https://keras.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter)](https://jupyter.org/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sarvesh1karandikar/Offspring-Face-Generator/blob/master/DCGAN_V2.ipynb)

---

## What This Does

Given a photo of a father and a photo of a mother, this model attempts to generate a plausible face image of their child. It is a research-grade deep learning project that explores the question: _can a neural network learn the visual genetics encoded in family photos?_

The model takes two 64×64 pixel face images as input — one for each parent — and produces a single 64×64 face image as output. It learns from real family photo datasets containing labelled father, mother, and child groupings.

---

## Architecture / Approach

The project explores two main approaches:

### 1. Encoder-Decoder (Primary Approach — `DCGAN_V2.ipynb`, `encoding-keras-2.0.ipynb`)

A supervised encoder-decoder neural network takes both parent images as input simultaneously:

- **Dual encoder branches**: Two parallel CNN stacks (each with strided Conv2D + BatchNorm + ELU layers) process the father and mother images separately, compressing each from 64×64×3 down to a compact feature map.
- **Feature fusion**: The two encoded feature maps are concatenated along the channel axis, merging the genetic information from both parents.
- **Decoder / generator**: A stack of transposed convolutions (Conv2DTranspose + BatchNorm + ReLU) upsamples the fused representation back to a 64×64 RGB image, with a final `tanh` activation to produce pixel values in [-1, 1].
- **Loss**: Pixel-wise L1 loss between the generated image and the actual child photo guides training.

### 2. DCGAN / WGAN-GP Baseline (TF1 — `model.py`, `generator.py`, `discriminator.py`)

An older TensorFlow 1.x implementation of a Deep Convolutional GAN with optional WGAN-GP, LSGAN, or hinge loss. The generator starts from a random noise vector (128-dim), projects it through a fully connected layer and reshape, then upsamples through bilinear deconvolution blocks with residual connections to produce 224×224×3 images. The discriminator mirrors this with strided convolutions and optional residual blocks.

### 3. Pre-GAN Pretraining (`Transfer_Learning.ipynb`)

A smaller GAN is first trained on child face images alone (without parent conditioning) to learn the distribution of child faces. This pre-trained generator is then used as a warm start for the full conditional generation task — a form of transfer learning designed to speed up convergence.

### Datasets

- **OFG Family dataset** (`datasets/ofg_family/`): Organised into family folders, each containing subdirectories for `father/`, `mother/`, `child_male/`, and `child_female/`.
- **TSKinFace dataset** (`datasets/TSKinFace_Data/TSKinFace_cropped/`): A standard kinship verification benchmark providing father (F), mother (M), son (S), and daughter (D) face triplets.

---

## Results

The model outputs 64×64 RGB face images. During evaluation:

- A side-by-side plot shows the father's face, the mother's face, the model's generated child face, and the actual child photo.
- Training loss curves (generator loss and discriminator loss) are plotted after each epoch to monitor convergence.
- Checkpoints are saved per epoch so intermediate generations can be compared.

Qualitatively, the generated faces tend to blend coarse facial structure (face shape, skin tone) from both parents, though fine-grained features like specific eye shape or nose structure are softened — an expected limitation of L1-trained encoder-decoders, which produce slightly blurry outputs.

---

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

### Run the Main Notebook

Open `DCGAN_V2.ipynb` in Jupyter or Google Colab:

```bash
jupyter notebook DCGAN_V2.ipynb
```

Or click the Colab badge at the top of this README.

### Data

Place your datasets in the following structure before running:

```
datasets/
  ofg_family/
    <family_id>/
      father/   *.jpg
      mother/   *.jpg
      child_male/   *.jpg   (or child_female/)
  TSKinFace_Data/
    TSKinFace_cropped/
      FMS/   FMS-<n>-F.jpg, FMS-<n>-M.jpg, FMS-<n>-S.jpg
      FMD/   ...
      FMSD/  ...
```

### Training the DCGAN Baseline (TF1)

```bash
python trainer.py --dataset OFG_Family --gan_type wgan-gp --batch_size 32 --img_h 224 --img_w 224
```

### Evaluation

```bash
python evaler.py --train_dir ./train_dir/<run_name> --write_summary_image True
```

---

## Portfolio Note: What a Live Demo Would Look Like

A fully interactive demo would work as follows:

1. **User uploads two photos** — one of the father, one of the mother — via a browser UI.
2. **Face alignment** is applied automatically (dlib or OpenCV) to crop and resize to 64×64.
3. **The model runs inference** (ideally converted to ONNX or TFLite for in-browser speed) and returns the predicted child face in under a second.
4. **Output panel** displays the father, mother, and generated child face side-by-side.

This could be built with a Streamlit front-end (< 1 day of work) and hosted on Hugging Face Spaces for free, making it immediately shareable with recruiters and colleagues.

---

## What I Learned / Key Insights

- **Conditioning on multiple inputs is non-trivial**: Naively concatenating parent images as a 6-channel input works but loses spatial correspondence. A dual-encoder with late feature fusion (as implemented here) gives the model more flexibility to learn parent-specific features before merging.
- **L1 loss produces blurry outputs**: Pixel-wise reconstruction loss averages over plausible outputs, resulting in slightly blurry generated faces. Adding an adversarial discriminator loss (as in the GAN pipeline) sharpens textures at the cost of training stability.
- **Transfer learning accelerates GAN training**: Pre-training a small unconditional GAN on child faces before adding the parent-conditioning branches provided a better weight initialisation and reduced the number of epochs needed for the conditional model to produce recognisable faces.
- **Dataset quality matters more than model complexity**: The OFG and TSKinFace datasets are relatively small (~hundreds of families). The bottleneck for improving quality is data quantity and face alignment quality, not model architecture.
- **WGAN-GP stabilises GAN training**: The Wasserstein loss with gradient penalty (implemented in `model.py`) provided more stable training curves compared to vanilla BCE or LSGAN losses on this small dataset.

---

## File Structure

```
Offspring-Face-Generator/
├── DCGAN_V2.ipynb              # Main notebook: dual-encoder conditional GAN (TF2/Keras)
├── encoding-keras-2.0.ipynb    # Encoder-decoder experiments
├── Transfer_Learning.ipynb     # Pre-GAN pretraining on child faces
├── DCGAN.ipynb                 # Earlier DCGAN experiments
├── DCGAN-TSkin_Old.ipynb       # Older TSKinFace experiments
├── test_encoding.ipynb         # Inference / evaluation notebook
├── CleanupCode.ipynb           # Data cleaning utilities
├── model.py                    # TF1 GAN model (Generator + Discriminator)
├── generator.py                # Generator class (TF1)
├── discriminator.py            # Discriminator class (TF1)
├── trainer.py                  # Training loop (TF1)
├── evaler.py                   # Evaluation loop (TF1)
├── ops.py                      # Custom layers: conv2d, deconv, residual blocks
├── input_ops.py                # Dataset batching pipeline
├── config.py                   # Argument parser and model factory
├── script_sort_images.py       # Data preprocessing utility
├── datasets/                   # Dataset directories (not included in repo)
└── requirements.txt
```

---

## License

MIT License — see [LICENSE](LICENSE).
