# From Pixels to Privacy: Temporally Consistent Video Anonymization via Token Pruning for Privacy Preserving Action Recognition
---

[Nazia Aslam](https://rabusi.github.io/), [Abhisek Ray](https://scholar.google.com/citations?user=a7HOeC8AAAAJ&hl=en), [Joakim Bruslund Haurum](https://scholar.google.com/citations?hl=en&user=GAEtgr4AAAAJ&view_op=list_works&sortby=pubdate), [Lukus Esterle](https://scholar.google.com/citations?user=KOC8OykAAAAJ&hl=en), [Kamal Nasrollahi](https://scholar.google.com/citations?user=EqjkO6sAAAAJ&hl=en)


[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2603.26336)

⭐ If you find this work helpful to your research, Don't forget to give a star to this repo. Thanks! 🤗

---

**Official PyTorch implementation for From Pixels to Privacy: Temporally Consistent Video Anonymization via Token Pruning for Privacy Preserving Action Recognition.**

> **Abstract:**
Recent advances in large-scale video models have significantly improved video understanding across domains such as surveillance, healthcare, and entertainment. However, these models also amplify privacy risks by encoding sensitive attributes, including facial identity, race, and gender. While image anonymization has been extensively studied, video anonymization remains relatively underexplored, even though modern video models can leverage spatiotemporal motion patterns as biometric identifiers.
To address this challenge, we propose a novel attention-driven spatiotemporal video anonymization framework based on the systematic disentanglement of utility and privacy features. Our key insight is that attention mechanisms in Vision Transformers (ViTs) can be explicitly structured to separate action-relevant information from privacy-sensitive content. Building on this insight, we introduce two task-specific classification tokens: an **action CLS token** and a **privacy CLS token**, which learn complementary representations within a shared Transformer backbone. We contrast their attention distributions to compute a utility–privacy score for each spatiotemporal tubelet and retain the top-*k* tubelets with the highest scores. This selectively prunes tubelets dominated by privacy cues while preserving those most critical for action recognition. Extensive experiments demonstrate that our approach maintains action recognition performance comparable to models trained on raw videos, while substantially reducing privacy leakage. These results indicate that attention-driven spatiotemporal pruning offers an effective and principled solution for privacy-preserving video analytics.

---

## 🏗️ Project Structure

```text
From-Pixels-to-Privacy/
├── action_training/
├── privacy_training/
├── sparsification/
├── aux_code/
├── annotations/
├── images/
├── conda_requirements.txt
├── pip_requirements.txt
├── vid_anon.yml
└── LICENSE
```

---

## 🧩 Proposed Framework

<p align="center">
  <img src="images/fig.jpg" alt="Architecture" width="900">
</p>

---

## Anonymized Images

<p align="center">
  <img src="images/fig_1.jpeg" alt="Weight lifting" width="100%">
  <img src="images/fig_2.png" alt="YoYo" width="100%">
  <img src="images/fig_11.png" alt="Playing violin" width="100%">
</p>

---

## 📊 Results

<p align="center">
  <img src="images/Table1.jpg" alt="Results Table 1" width="100%">
</p>

<p align="center">
  <img src="images/Table2.jpg" alt="Results Table 2" width="100%">
</p>

---

## 📬 Contact

For any inquiries or feedback, feel free to reach out:

- **Nazia Aslam**
- **Email:** [naas@create.aau.dk](mailto:naas@create.aau.dk)

---
