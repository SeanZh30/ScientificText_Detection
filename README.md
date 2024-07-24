# ScientificText_Detection

# Nuanced Multi-class Detection of Machine-Generated Scientific Text

## Overview

This repo is for share datasets generated from paper: Nuanced Multi-class Detection of Machine-Generated Scientific Text

**Abstract**:
Recent advancements in large language models (LLMs) have demonstrated their capacity to produce coherent scientific text, often indistinguishable from human-authored content. However, this raises significant concerns regarding the potential misuse of such techniques, posing threats to research advancement across various domains. In this study, we focus on nuanced detection of machine-generated scientific text and build a new multi-domain dataset for this task. Instead of treating the detection as binary classification as previous work, we additionally consider the classification of diverse practical usages of LLMs, including paraphrasing, summarization, and title-based generation. Additionally, we introduce a novel baseline model integrating contrastive learning, encouraging the model to discern similar text more effectively. Experimental results underscore the efficacy of our proposed method compared to prior baselines, supplemented by an analysis of domain generalization conducted on our dataset.

## Dataset

We provide access to our multi-domain dataset designed for nuanced detection of machine-generated scientific text. The dataset is divided into training, validation, and test sets.

### Full Dataset
[Full Dataset Link](https://drive.google.com/file/d/1_F7kpRdfPhfMp1gfbe14FaENUe0zSy6i/view?usp=share_link)

### Validation Dataset
[Validation Dataset Link](https://drive.google.com/file/d/1KF_Xn9HTtKRlx3SXrJu097054Nj3X2tG/view?usp=share_link)

### Test Dataset
[Test Dataset Link](https://drive.google.com/file/d/1Gqb7KVMs557m1GQFG0QpZk5HpMB1AD8n/view?usp=share_link)

### Train Dataset
[Train Dataset Link](https://drive.google.com/file/d/149QXLsQ2R_Ivh5K-zAfZ7P6stQGNtIDm/view?usp=share_link)

## Labels

The dataset includes the following labels for classification:

- `0` - Generated
- `1` - Paraphrase
- `2` - Summarized
- `3` - Human Write

### Labeled Dataset
[Labeled Dataset Link](https://drive.google.com/drive/folders/1bgO4hmqKKOLb3Eqbbjml848YB7tCHirR?usp=share_link)

## Usage

To use the dataset, download the relevant files from the links provided above. The dataset is organized into separate folders for training, validation, and testing. Each instance in the dataset is labeled according to the categories mentioned above.

## Citation

If you use this dataset in your research, please cite our paper:

```
@article{yourpaper2024,
  title={Nuanced Multi-class Detection of Machine-Generated Scientific Text},
  author={Your Name and Co-authors},
  journal={Journal Name},
  year={2024},
  volume={Volume},
  pages={Page numbers},
  publisher={Publisher}
}
```

## Contact

For any questions or issues related to the dataset, please contact:

- **Yubin Ge**: [yubinge2@illinois.edu](yubinge2@illinois.edu)
- **Shiyuan Zhang**: [sz54@illinois.edu](sz54@illinois.edu)
