# CLIP behaves like a bag-of-words model cross-modally but not uni-modally

This repository contains code for [CLIP behaves like a bag-of-words model cross-modally but not uni-modally](https://arxiv.org/abs/2502.03566).

**TL;DR**: CLIP behaves like a bag-of-words cross-modally, but we show that attribute-object binding exists within its embeddings using linear probing. A simple linear transformation aligns these signals and enhances cross-modal binding.

## Installation

1. Clone the repository:
   
   ```bash
   git clone https://github.com/kdariina/CLIP-not-BoW-unimodally.git
   cd CLIP-not-BoW-unimodally
   ```

2. Install dependencies:
   
   ```bash
   pip install -r requirements.txt
   ```

3. We use code from *When and why vision-language models behave like bags-of-words, and what to do about it?* by Yuksekgonul et al. (2023). Clone `vision-language-models-are-bows` and place it inside this directory:
   
   ```bash
   git clone https://github.com/mertyg/vision-language-models-are-bows.git
   ```

## Repository structure

- `datasets/` – Contains datasets used in experiments:
  - `ARO/`, `COCO/`, `CLEVR/`, `PUG_SPAR/`, `PUG_SPARE/`, `SugarCrepe/` (with subdirectories for images and metadata).
- `clevr_generation/` – Code for generating CLEVR images.
- `data_loading/` – Scripts for loading datasets and extracting embeddings.
  - `pug.py` – Handles PUG:SPAR and PUG:SPARE datasets.
  - `clevr.py` – Handles CLEVR dataset.
- `cache/` – Stores precomputed embeddings and trained models.
- `probing/` – Code for linear probing experiments:
  - `probing_datasets.py` – Prepares dataloaders for linear probing.
  - `probing_models.py` – Implements probing models.
  - `pug_probing.py`, `clevr_probing.py`, `clevr_multi_object_probing.py` – Scripts for running probing experiments.
- `alignment/` – Code for LABCLIP implementation and evaluation:
  - `alignment_datasets.py` – Prepares dataloaders for LABCLIP.
  - `learning_alignment.py` – Contains the LABCLIP model, losses, and training procedure.
  - `pug_alignment.py`, `clevr_alignment.py`, `coco_alignment.py` – Runs LABCLIP for these datasets.
  - `ARO.ipynb`, `sugarcrepe_eval.ipynb` – Evaluation notebooks for ARO and SugarCrepe.

## Datasets

The repository uses several datasets:

- [**CLEVR**](https://cs.stanford.edu/people/jcjohns/clevr/) – A dataset featuring simple shapes and colors generated with Blender. We adapt the code to produce images for our project. You can download them from [Google Drive](https://drive.google.com/drive/folders/1AC5Xv7-vTG6coATBZKnaaNFbTzqkW-4C?usp=sharing) or use `clevr.py` to handle it. You can also generate your own images with scripts in `clevr_generation/`.
- [**PUG:SPAR**](https://pug.metademolab.com/) – A synthetic dataset created in Unreal Engine, featuring animal figures on various backgrounds.
- ✨**PUG:SPARE**✨ – A modified version of PUG:SPAR that mitigates positional shortcuts. Download it from [Google Drive](https://drive.google.com/drive/folders/19NLxbY_ns8hWNrdhNdcb7oF1RbsYU1jt?usp=sharing) or let `pug.py` download the files automatically.
- [**COCO**](https://cocodataset.org/#download) – A large-scale dataset with diverse scenes and object categories. We use the COCO 2014 version with Karpathy splits: [train](https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_train.json), [val](https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json), [test](https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json).
- [**ARO**](https://github.com/mertyg/vision-language-models-are-bows) – A benchmark that tests compositionality in VLMs using real-world images.
- [**SugarCrepe**](https://github.com/RAIVNLab/sugar-crepe) – A benchmark for evaluating compositional understanding in VLMs by generating fluent and sensible hard negatives.

> PUG:SPARE was created in Unreal Engine using free assets from the Unreal Engine Marketplace that are marked as NoAI Content. This dataset was used solely for analysis and not for training, developing, or generating content with Generative AI. Per Unreal Engine’s licensing terms, this dataset must not be used for Generative AI programs, the development of Generative AI, or as input to Generative AI models. 

> Some datasets require separate downloads. Ensure dataset paths are correctly specified when running scripts.

## Usage

### Running probing experiments

To run probing experiments, navigate to:

```bash
cd probing
```

To run probing on PUG_SPAR or PUG_SPARE datasets:

```bash
python pug_probing.py --dataset PUG_SPARE --probe_type image [other_args]
```

To run probing on CLEVR for the standard 2-object case:

```bash
python clevr_probing.py --data_path ../datasets/CLEVR --probe_type image [other_args]
```

For multi-object experiments on CLEVR:

```bash
python clevr_multi_object_probing.py --data_path ../datasets/CLEVR --probe_type image --num_objects 3 [other_args]
```

For additional arguments, refer to the corresponding Python files.

### Linear Attribute Binding CLIP (LABCLIP)

The code for LABCLIP training and evaluation is in the `alignment/` directory:

```bash
cd alignment
```

To train and evaluate LABCLIP on PUG:SPAR or PUG:SPARE:

```bash
python pug_alignment.py --dataset PUG_SPARE --alignment_type HNB [other_args]
```

To train and evaluate LABCLIP on CLEVR:

```bash
python clevr_alignment.py --alignment_type HNB [other_args]
```

To train and evaluate LABCLIP on COCO:

```bash
python coco_alignment.py --data_path ../datasets/COCO --alignment_type HNB [other_args]
```

## Citation

If you use this code or the PUG:SPARE dataset, please cite:

```bibtex
@article{koishigarina2025_2502.03566,
  title={CLIP behaves like a bag-of-words model cross-modally but not uni-modally},
  author={Darina Koishigarina and Arnas Uselis and Seong Joon Oh},
  journal={arXiv preprint arXiv:2502.03566},
  year={2025}
}
```

## Contact

For questions and requests, please reach out to Darina at **darina51012@gmail.com**.

