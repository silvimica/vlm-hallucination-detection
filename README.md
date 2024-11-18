# VLMs-Hallucination-Detection

This repository contains code and resources for detecting hallucinations in Vision-Language Models (VLMs) using only language data. The goal is to identify instances where VLMs generate text that is not grounded in the input data, thus reducing the likelihood of hallucinated or inaccurate responses.

## Table of Contents

- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Installation](#installation)

## Project Overview

Vision-Language Models (VLMs) can sometimes generate hallucinated text that isn't properly grounded in the provided visual or textual data. This project aims to build a framework for detecting such hallucinations by analyzing language outputs from VLMs without relying on the image data. It provides methods for dataset preparation, model training, and evaluation.

## Directory Structure
```plaintext
├── data/           # Contains the datasets used for training and evaluation
├── notebooks/      # Jupyter notebooks for data exploration and model performance demonstration
├── src/            # Source code for utility functions
└── README.md       # Project documentation
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/silvimica/vlm-hallucination-detection.git
   cd vlm-hallucination-detection
   ```

2. Create a virtual environment with python==3.10 (optional but recommended).

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
