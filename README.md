# Toy-Diffusion

**Toy-Diffusion** is a research sandbox designed to ablate and visualize
different diffusion modeling approaches, specifically comparing classical
DDPM (SDE) with modern Flow Matching (ODE) objectives.

This repository allows for tractable testing of loss functions, sampling trajectories, and
parametrizations ($x_0$, $\epsilon$, or $v$ prediction).

## 🛠️ Key Functionalities

### 1. 2D Synthetic Manifolds
Train models on complex 2D distributions including:
*   **Gaussian Mixtures (GMM)**: Test mode collapse and imbalanced clusters.
*   **Kanji (あ/お)**: High-resolution continuous distributions from glyphs.
*   **Spirals & Pinwheels**: Evaluate path straightness and curvature.
*   **Manifold Projection**: Test models in high-dimensional spaces (up to 512D)
    projected back to 2D for evaluation.

### 2. Pixel & Latent Image Training
A streamlined pipeline for image datasets:
*   **Latent Support**: Pre-encode images using VAEs (e.g., FLUX or SD1.5) and
    train directly on `.npz` latents.
*   **Empirical Normalization**: Automatically calculates latent mean/std for
    optimal scaling ($v = x_1 - x_0$ is sensitive to variance).
*   **Tiered Batching**: Efficiently handles different sequence lengths for
    conditional tag-based training.

### 3. Interactive Visualization UI
A Gradio-based dashboard to "feel" the vector fields:
*   **Vector Field Plotting**: Visualize marginal velocities and scores.
*   **Particle Integration**: Click anywhere on the distribution to spawn a
    particle and watch it flow through the learned ODE.
*   **Schedule Comparison**: Real-time switching between Linear (FM) and
    DDPM paths.

### 4. Inference
The pretrained models can be used for inference in 2 ways.
*   **Gradio UI**: dashboard to generate images using custom prompts.
*   **CLI**: script to generate samples using the terminal.

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/deeplearningcafe/toy-diffusion.git
cd toy-diffusion

# Install dependencies
pip install -r requirements.txt

# Install as editable package
pip install -e .
```

## 🚀 Usage

### Interactive UI
Recommended for initial experimentation with 2D physics.
```bash
python scripts/main.py --config configs/toy_example.yaml
```

### Training Anime Faces (Latents)
To run the experiment for shifting distributions to modern anime styles:
1. Encode your images:
   ```bash
   python scripts/encode_latents.py --data_dir ./data/anime --half
   ```
2. Launch training:
   ```bash
   python scripts/anime_faces.py --config configs/toy_example.yaml
   ```

### Ablation: Flow Matching vs DDPM
Compare how fast a model learns a new distribution (e.g., Kanji) when
switching from $\epsilon$-prediction to $v$-prediction:
```bash
python scripts/finetune.py --config configs/toy_example.yaml \
    data.dataset_type=kanji training.epochs=100
```

## 🔬 Repository Structure

*   `src/toy_diffusion/paths/`: The core physics engine. Contains
    `scheduler.py` (alpha/sigma curves) and `sampling.py` (ODE/SDE solvers).
*   `src/toy_diffusion/models/`: Architectures including `Unet` for
    images and `FlowMLP` for 2D data.
*   `src/toy_diffusion/losses.py`: Unified loss class that handles the
    algebraic inversion of targets (e.g., calculating $v$-loss from an
    $\epsilon$-prediction model).
*   `scripts/`: Entry points for experiments, profiling, and evaluation.

---

### Main files and classes

The core logic resides in `src/toy_diffusion/losses.py` and
`src/toy_diffusion/trainer.py`.

1.  **`GeneralDiffusionLoss` (`losses.py`)**: This is the most critical class.
    It implements the "Unified Perspective." It takes a model's raw output
    (which could be $x$, $\epsilon$, or $v$) and uses the `Wronskian`
    determinant of the schedule to convert that prediction into the desired
    `loss_target`. This allows us to train a $v$-prediction model using an
    $\epsilon$-loss, or vice versa, to study gradient stability.
2.  **`Trainer` (`trainer.py`)**: This class manages the lifecycle. It
    implements **Activation Checkpointing** and **EMA (Exponential Moving
    Average)**. EMA is particularly vital in Flow Matching because the
    straight-path trajectories can be noisy early in training; the EMA
    weights provide the stability needed for coherent sampling during
    evaluation.
3.  **`ImageDataset` (`data/image.py`)**: It implements a **Tiered Sampler**.
    Since anime tags vary in length, it groups prompts into "tiers" (e.g.,
    length 24, length 52) to minimize padding tokens in the `SimpleTextEncoder`,
    maximizing TFLOPS during the cross-attention layers.

## References
*   **Danbooru2024 Dataset**: [p1atdev/danbooru-2024](https://huggingface.co/datasets/p1atdev/danbooru-2024)
*   **SnapGen**: [paper](https://arxiv.org/abs/2412.09619). The `EfficientUnet` model implementation is based on this paper.

## Author
[aipracticecafe](https://github.com/deeplearningcafe)
[aipracticecafe-codeberg](https://codeberg.org/aipracticecafe)

## License
This project is licensed under the `MIT`. Details are in the [LICENSE](LICENSE.txt) file. I don't own the data from the `danbooru` website.
