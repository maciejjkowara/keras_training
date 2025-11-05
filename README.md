# Keras Multi-Output Classification - Bond Data

Learning exercise using Keras Functional API to predict bond classifications (level_2 and level_3) based on bond characteristics.

## Open in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maciejjkowara/keras_training/blob/main/index_constitutents_classification.ipynb)

## Project Overview

This notebook demonstrates building a multi-output neural network using Keras Functional API to predict:
- **level_2**: Asset class (Securitized vs Industrials)
- **level_3**: Sector classification (Commercial Mortgage B, Healthcare, Basic Industry, Technology & Electronics)

### Input Features
- `oas`: Option-adjusted spread
- `yield`: Bond yield
- `duration`: Duration measure
- `convexity`: Convexity measure
- `rating`: Credit rating (categorical)
- `coupon`: Coupon rate

### Tech Stack
- Keras 3 with JAX backend
- Functional API for multi-output architecture
- scikit-learn for preprocessing

## Dataset

10,000+ bond observations with classification targets and numerical/categorical features.
