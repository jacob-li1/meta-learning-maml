# Meta-Learning Implementation (MAML, FOMAML, ANIL, Reptile)

This project implements several classical meta-learning algorithms for few-shot classification using PyTorch.

## 📌 Algorithms Implemented

- MAML (Model-Agnostic Meta-Learning)
- FOMAML (First-Order MAML)
- ANIL (Almost No Inner Loop)
- Reptile

---

## 📊 Experimental Results

| Algorithm | Accuracy |
|----------|--------|
| MAML | 62.3% |
| FOMAML | 52.8% |
| ANIL | 79.2% |
| Reptile | 33.8% |

> Note: Results are based on Omniglot few-shot classification tasks.

---

## 🧠 Key Features

- Functional forward implementation (no in-place parameter updates)
- Unified meta-learning framework
- Support for different inner/outer update strategies
- Modular design for easy algorithm comparison

---

## 📂 Project Structure
```
.
├── ML2021_HW15_Meta_Learning.ipynb # main implementation
├── README.md
└── .gitignore
```

---

## ⚙️ Requirements

Install dependencies:

```bash
pip install torch torchvision numpy
```

---

## 🚀 How to Run

Open the notebook and run all cells:
```
jupyter notebook ML2021_HW15_Meta_Learning.ipynb
```

---

## 📦 Dataset

This project uses the Omniglot dataset.

⚠️ Dataset is NOT included in this repository.

You can download it from:
https://github.com/brendenlake/omniglot

---

## 📝 Notes

- MAML uses higher-order gradients (if enabled)
- FOMAML ignores second-order derivatives
- ANIL only updates the final layer in inner loop
- Reptile performs meta-update via parameter difference

