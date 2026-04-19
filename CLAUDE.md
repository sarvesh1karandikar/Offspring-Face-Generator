# CLAUDE.md — Offspring Face Generator

## What This Project Is

Offspring Face Generator is a deep learning research project that trains neural networks to predict what a child's face might look like given photos of both biological parents. It uses a dual-encoder convolutional architecture built in TensorFlow 2/Keras: two CNN branches encode the father's and mother's face images independently into compact feature vectors, which are then fused and decoded by a transposed-convolution generator to produce a 64×64 RGB child face image. The project also includes an older TensorFlow 1.x DCGAN pipeline (WGAN-GP / LSGAN / hinge loss variants) and a pre-training step where a small unconditional GAN is trained on child-only faces before introducing parent conditioning.

---

## How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Open the main notebook

```bash
jupyter notebook DCGAN_V2.ipynb
# or
jupyter notebook encoding-keras-2.0.ipynb
```

### Run the TF1 GAN trainer (requires TF 1.x)

```bash
python trainer.py \
  --dataset OFG_Family \
  --gan_type wgan-gp \
  --batch_size 32 \
  --img_h 224 \
  --img_w 224 \
  --learning_rate_g 1e-4 \
  --learning_rate_d 1e-4
```

### Run evaluation

```bash
python evaler.py \
  --train_dir ./train_dir/<run_name> \
  --write_summary_image True \
  --summary_image_name eval_output.png
```

### Data structure required

```
datasets/ofg_family/<family_id>/{father,mother,child_male,child_female}/
datasets/TSKinFace_Data/TSKinFace_cropped/{FMS,FMD,FMSD}/
```

---

## Model Architecture Summary

### TF2 Dual-Encoder Conditional Generator (`DCGAN_V2.ipynb`, `encoding-keras-2.0.ipynb`)

- **Input**: Two 64×64×3 RGB images (father, mother) as separate Keras Input tensors.
- **Encoder (shared weights per parent)**: 2–3 stacked `Conv2D` layers with stride 2, filter sizes [32, 64], kernel size 4, BatchNorm, ELU activation. Reduces 64×64 → 16×16 feature maps.
- **Fusion**: `Concatenate` along channel axis merges the two encoded feature maps.
- **Decoder**: `Conv2DTranspose` layers with stride 2, filter sizes [32, OUTPUT_CHANNELS], BatchNorm, ReLU. Final layer uses `tanh` activation.
- **Loss**: L1 pixel-wise reconstruction loss against the real child image.
- **Optimizer**: Adam (lr=5e-4, β1=0.875, β2=0.975).

### TF1 DCGAN / WGAN-GP (`model.py`, `generator.py`, `discriminator.py`)

- **Generator**: FC layer → reshape to 4×4×256 → N bilinear deconv blocks (doubling spatial dims, halving channels) → residual blocks at later layers → final 1×1 conv with tanh. Outputs 224×224×3.
- **Discriminator**: 6 strided conv layers (32→64→128→256→256→512 channels) + residual blocks + 1×1 conv head. Produces a spatial real/fake map.
- **GAN losses**: Supports WGAN-GP (default, γ=10), LSGAN, or hinge loss.
- **Noise dim**: 128.
- **Normalization**: Batch norm in generator, none in discriminator (standard for WGAN-GP).

### Pre-GAN (`Transfer_Learning.ipynb`)

- Unconditional GAN trained on child face images only (64×64).
- Generator: noise input 4×4×64 → 3× `Conv2DTranspose` upsample → tanh output.
- Discriminator: 4× strided `Conv2D` → sigmoid output.
- Weights used to warm-start the conditional model.

---

## Current Limitations

1. **No pretrained weights in repo** — the repo does not include any saved model checkpoints, so training must be run from scratch, which requires GPUs and the datasets.
2. **Datasets not included** — both `ofg_family` and `TSKinFace_Data` are large external datasets that must be downloaded separately and are not committed to the repo.
3. **Low output resolution** — encoder-decoder notebooks generate 64×64 images; the TF1 pipeline targets 224×224 but is harder to run (requires TF 1.x).
4. **Blurry outputs** — L1 loss in the encoder-decoder produces smooth but blurry child faces. No perceptual loss or GAN discriminator is used in the TF2 notebooks.
5. **TF1 / TF2 split** — the codebase is split across TF1 (model.py, trainer.py) and TF2/Keras (notebooks), making it difficult to run end-to-end without managing two environments.
6. **No face alignment** — raw images are simply resized to 64×64 without landmark-based alignment, so head pose and scale variation adds noise.
7. **Small datasets** — hundreds to low thousands of family triplets limit generalisation.
8. **No quantitative evaluation** — no FID, SSIM, or kinship verification metric is computed in the notebooks.

---

## TODO: Making This Demo-able on a Portfolio Website

### Data / Model

- [ ] Train the encoder-decoder model to convergence and export weights as `.h5` or `.keras` file
- [ ] Add face alignment (OpenCV + dlib 68-landmark crop) as a preprocessing step
- [ ] Upgrade the DCGAN pipeline to TF2 / PyTorch to unify the codebase
- [ ] Add perceptual loss (VGG feature matching) to reduce blurriness
- [ ] Increase output resolution to 128×128 or 256×256 via progressive upsampling

### Deployment

- [ ] Convert the trained Keras model to ONNX (`tf2onnx`) for lightweight browser or server inference
- [ ] Alternatively, convert to TFLite for mobile/edge deployment
- [ ] Build a Streamlit app (`app.py`) with:
  - File upload widgets for father and mother photos
  - Automatic face detection and crop (OpenCV Haar cascade or MTCNN)
  - Model inference call
  - Side-by-side display: father | mother | generated child
- [ ] Host on Hugging Face Spaces (Streamlit SDK, free tier)
- [ ] Add a `gradio` demo as an alternative (1-file, zero-config)

### Repository Hygiene

- [ ] Add sample output images (generated child faces) to README
- [ ] Add a `demo/` folder with example parent image pairs
- [ ] Write a `setup.py` or `pyproject.toml`
- [ ] Add GitHub Actions CI to lint notebooks with `nbconvert --execute`

---

## Recommended Demo Tier: Medium Lift

**Justification**: The core model architecture is already implemented and working — the encoder-decoder notebook is self-contained and runnable. The missing pieces are (a) trained weights and (b) a thin Streamlit/Gradio wrapper, which together represent roughly 1–2 days of engineering. This is not a quick win because weights must be produced by training (needs GPU time and data), but it is significantly less than a "big lift" because no new model design work is needed — just training, exporting, and wrapping in a UI.

A Quick Win is not possible here because no inference-ready weights are committed to the repo, so the demo cannot be assembled without a training run. A Big Lift would be warranted if the model needed fundamental architectural improvements (e.g. upgrading to a diffusion model or StyleGAN-based approach) to produce results good enough to show publicly.
