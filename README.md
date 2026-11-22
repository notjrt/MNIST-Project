# Handwritten Digit Classification on MNIST: A Comparative Study of Machine Learning Methods

A machine learning project investigating handwritten digit classification using the MNIST dataset. The study evaluates **k-Nearest Neighbours (kNN)**, training-set reduction techniques, **Logistic Regression**, and **Support Vector Machines (SVM)**.

## Project Overview

MNIST is a benchmark dataset containing 70,000 grayscale images of handwritten digits (0–9). Each image has a resolution of 28 × 28 pixels.

The dataset is divided into:

| Dataset | Images |
|---|---:|
| Training | 60,000 |
| Testing | 10,000 |
| Classes | 10 |
| Features | 784 |

Pixel intensities are normalised from `[0, 255]` to `[0, 1]`.

The project comprises three experiments:

1. **kNN Optimisation:** Investigating the influence of the number of neighbours on classification accuracy.
2. **Data Reduction:** Implementing preliminary cleaning and Condensed Nearest Neighbour (CNN), and examining their effects using PCA.
3. **Algorithm Comparison:** Comparing kNN against Logistic Regression and RBF-SVM.

---

## Repository Structure

```text
MNIST-Project/
├── notebooks/
│   ├── MNIST_Experiment1.ipynb
│   ├── MNIST_Experiment2.ipynb
│   └── MNIST_Experiment3.ipynb
│
├── results/
│   ├── experiment1/
│   ├── experiment2/
│   └── experiment3/
│
├── report/
│   ├── final_paper.tex
│   └── final_paper.pdf
│
├── experiments.py
├── requirements.txt
├── .gitignore
└── README.md
```

### Notebooks

- **Experiment 1:** kNN optimisation using three-fold cross-validation.
- **Experiment 2:** Lecture-based data reduction using preliminary cleaning, CNN and PCA.
- **Experiment 3:** Comparison of kNN, Logistic Regression and RBF-SVM.

### Results

Each experiment has a dedicated folder containing its generated figures, evaluation results and any relevant CSV files.

### Report

The `report/` directory contains the final academic report written in LaTeX using the ICML 2026 format.

---

## Experiment 1 — kNN Optimisation

### Objective

Determine the best value of `k` for handwritten digit classification using three-fold stratified cross-validation.

### Methodology

- Use Euclidean distance.
- Evaluate `k = [1, 3, 5, 7, 9]`.
- Perform three-fold stratified cross-validation on the 60,000 training images.
- Select the highest-scoring configuration.
- Evaluate the selected model on the independent 10,000-image test set.

### Results

| k | Mean Validation Accuracy | Standard Deviation |
|---|---:|---:|
| 1 | 97.0117% | 0.0685% |
| **3** | **97.0167%** | 0.1684% |
| 5 | 96.8950% | 0.1350% |
| 7 | 96.7250% | 0.1806% |
| 9 | 96.5367% | 0.1336% |

**Selected model:** kNN with `k = 3`

**Test accuracy:** 97.05% (9,705 correctly classified images).

The difference between `k = 1` and `k = 3` was negligible. Accuracy generally decreased for larger neighbourhood sizes.

---

## Experiment 2 — Training-Set Reduction

### Objective

Investigate whether the number of stored training examples can be reduced while maintaining acceptable 1NN classification accuracy.

The algorithms are based on the *Introduction to Machine Learning — Part 3: k-Nearest Neighbors* lecture slides by **Marc Sebban**, Université Jean Monnet.

### Methods

**1. Preliminary Two-Subset Cleaning**

The training set is randomly divided into two subsets, S1 and S2.

- Classify S1 using S2 as the 1NN reference set.
- Remove misclassified examples from S1.
- Classify S2 using the updated S1.
- Remove misclassified examples from S2.
- Repeat until neither subset changes.

**2. Condensed Nearest Neighbour (CNN)**

CNN constructs a smaller reference dataset by retaining examples that the current 1NN reference set cannot correctly classify.

Our implementation begins with one example per digit class and incrementally adds misclassified examples until convergence.

**3. Cleaning + CNN**

The preliminary cleaning procedure is applied first, followed by CNN condensation.

### Results

For a controlled comparison, all methods use the same stratified subset of 2,000 MNIST training examples and the original 10,000-image test set.

| Method | Retained Samples | Reduction | Test Accuracy | Prediction Time |
|---|---:|---:|---:|---:|
| Original 1NN | 2,000 | 0.00% | 91.20% | 0.284 s |
| Cleaning | 1,755 | 12.25% | 89.90% | 0.239 s |
| CNN | 516 | 74.20% | 86.34% | 0.167 s |
| **Cleaning + CNN** | **321** | **83.95%** | **85.14%** | **0.124 s** |

The combined cleaning and CNN method achieved the greatest reduction, removing 1,679 of the original 2,000 training samples.

However, this came with a decrease of 6.06 percentage points in classification accuracy.

### PCA Analysis

Principal Component Analysis (PCA) was used to visualise the original and reduced datasets in two dimensions.

Both datasets were projected into the same PCA coordinate system to examine the distribution of retained examples.

The reduced dataset remained distributed across multiple regions of the original feature space, although substantial overlap between digit classes was visible.

**Limitation:** The CNN experiments were conducted on a 2,000-image training subset rather than the complete 60,000-image dataset. Results should therefore not be interpreted as full-dataset CNN performance.

---

## Experiment 3 — Machine Learning Algorithm Comparison

### Objective

Compare kNN against two alternative classifiers using classification accuracy and computational efficiency.

### Models

- **kNN:** `k = 3`, Euclidean distance
- **Logistic Regression:** L-BFGS solver, `C = 1.0`, maximum 1,000 iterations
- **SVM:** RBF kernel, `C = 1.0`, `gamma = "scale"`

All models are trained on the same 60,000 MNIST training images and evaluated on the same 10,000 test images.

### Results

| Algorithm | Test Accuracy | Training Time | Prediction Time |
|---|---:|---:|---:|
| kNN (k=3) | 97.05% | 0.13 s | 4.90 s |
| Logistic Regression | 92.61% | 5.69 s | **0.02 s** |
| **SVM (RBF)** | **97.92%** | 139.41 s | 26.06 s |

### Classification Errors

| Algorithm | Correct | Incorrect |
|---|---:|---:|
| kNN | 9,705 | 295 |
| Logistic Regression | 9,261 | 739 |
| SVM (RBF) | 9,792 | 208 |

**Key findings:**

- **SVM (RBF)** achieved the highest classification accuracy.
- **kNN** provided strong accuracy with minimal fitting time.
- **Logistic Regression** achieved the fastest prediction time, but lower accuracy.

Training and prediction times represent individual experimental runs and may vary depending on hardware and system conditions.

---

## Technologies Used

- Python 3.12
- Jupyter Notebook
- NumPy
- pandas
- Matplotlib
- scikit-learn
- Visual Studio Code
- Overleaf (LaTeX)

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/notjrt/MNIST-Project.git
cd MNIST-Project
```

### 2. Create a virtual environment

```bash
python3.12 -m venv .venv
```

Activate the environment.

**macOS/Linux:**

```bash
source .venv/bin/activate
```

**Windows:**

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib scikit-learn ipykernel jupyter
```

### 4. Run the notebooks

Open the repository in VS Code, install the Python and Jupyter extensions, and select the virtual environment as the notebook kernel.

Run the experiments in the following order:

1. Experiment 1 — kNN Optimisation
2. Experiment 2 — Data Reduction and PCA
3. Experiment 3 — Algorithm Comparison

The MNIST dataset is downloaded using `fetch_openml()` and requires an internet connection on the first run. Subsequent runs may use the locally cached dataset.

## Reproducibility

The experiments use the official MNIST 60,000/10,000 train-test split.

Stratified sampling and cross-validation use `random_state=42` where applicable.

The custom reduction algorithms rely on nearest-neighbour search from scikit-learn but implement their own sample-selection procedures.

Experiment 2 uses a fixed stratified subset of 2,000 training images for direct comparison.

Minor differences in runtime or results may occur across library versions and computing environments.

## Conclusions

The experiments demonstrate several trade-offs between machine learning accuracy, data requirements, and computational efficiency.

- kNN achieved 97.05% accuracy with three neighbours.
- Cleaning combined with CNN reduced a 2,000-example reference set by 83.95%, at the expense of classification accuracy.
- RBF-SVM achieved the highest accuracy at 97.92%, but with greater computational cost.
- Logistic Regression provided the fastest predictions.

These findings highlight the importance of considering both predictive performance and computational requirements when selecting machine learning algorithms.

## Authors

- **Jayden Tan** — [@notjrt](https://github.com/notjrt)
- **Yuhsaku Nakai** - [@aliks-kuro](https://github.com/aliks-kuro)

## References

1. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). *Gradient-Based Learning Applied to Document Recognition*. Proceedings of the IEEE, 86(11), 2278–2324.
2. Hart, P. (1968). *The Condensed Nearest Neighbor Rule*. IEEE Transactions on Information Theory, 14(3), 515–516.
3. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.
4. Sebban, M. *Introduction to Machine Learning: Part 3 — k-Nearest Neighbors*. Laboratoire Hubert Curien, Université Jean Monnet. Course lecture slides, 2026–2027.

---
