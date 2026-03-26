# From Pixels to Privacy: Temporally Consistent Video Anonymization via Token Pruning for Privacy Preserving Action Recognition

<p align="center">
  <a href="https://github.com/Rabusi/From-Pixels-to-Privacy-Temporally-Consistent-Video-Anonymization-via-Token-Pruning-for-Privacy-Pres/stargazers">
    <img src="https://img.shields.io/github/stars/Rabusi/From-Pixels-to-Privacy-Temporally-Consistent-Video-Anonymization-via-Token-Pruning-for-Privacy-Pres?style=social" alt="GitHub Stars">
  </a>
  <a href="https://github.com/Rabusi/From-Pixels-to-Privacy-Temporally-Consistent-Video-Anonymization-via-Token-Pruning-for-Privacy-Pres/network/members">
    <img src="https://img.shields.io/github/forks/Rabusi/From-Pixels-to-Privacy-Temporally-Consistent-Video-Anonymization-via-Token-Pruning-for-Privacy-Pres?style=social" alt="GitHub Forks">
  </a>
  <a href="https://arxiv.org/abs/2504.14301">
    <img src="https://img.shields.io/badge/arXiv-2504.14301-b31b1b.svg" alt="arXiv:2504.14301">
  </a>
</p>

<p align="center">
  ⭐ If you find this work useful for your research, please consider starring this repository.
</p>

---

## 📄 Abstract

Recent advances in large-scale video models have significantly improved video understanding across domains such as surveillance, healthcare, and entertainment. However, these models also amplify privacy risks by encoding sensitive attributes, including facial identity, race, and gender. While image anonymization has been extensively studied, video anonymization remains relatively underexplored, even though modern video models can leverage spatiotemporal motion patterns as biometric identifiers. To address this challenge, we propose a novel attention-driven spatiotemporal video anonymization framework based on systematic disentanglement of utility and privacy features. Our key insight is that attention mechanisms in Vision Transformers (ViTs) can be explicitly structured to separate action-relevant information from privacy-sensitive content. Building on this insight, we introduce two task-specific classification tokens, an action CLS token and a privacy CLS token, that learn complementary representations within a shared Transformer backbone. We contrast their attention distributions to compute a utility–privacy score for each spatiotemporal tubelet, and keep the top-k tubelets with the highest scores. This selectively prunes tubelets dominated by privacy cues while preserving those most critical for action recognition. Extensive experiments demonstrate that our approach maintains action recognition performance comparable to models trained on raw videos, while substantially reducing privacy leakage. These results indicate that attention-driven spatiotemporal pruning offers an effective and principled solution for privacy-preserving video analytics.

---

## 🏗️ Project Structure

From-Pixels-to-Privacy/
├── action_training/
├── privacy_training/
├── sparsification/
├── aux_code/
├── annotations/
├── conda_requirements.txt
├── pip_requirements.txt
├── stprivacy.yml


## 🧩 Proposed Framework

![Architecture](images/architecture_mod.jpg)

## 🖼️ Anonymized Images

![Weight lifting](images/fig_1.png)
![YoYo](images/fig_2.png)
![Playing violon](images/fig_11.png)

## 📊 Results
![](images/Table1.jpg)
![](images/Table2.jpg)

## 📬 Contact

For any inquiries or feedback, feel free to reach out: [naas@create.aau.dk](mailto:naas@create.aau.dk)
├── README.md
└── LICENSE
