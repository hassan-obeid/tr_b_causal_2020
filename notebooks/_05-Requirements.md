---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.3.3
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# <ins>Requirements</ins>


## What exactly does the product need to deliver to meet the needs of the end users / stakeholders?

## Do the end user’s / stakeholders have firm time deadlines for the project? If so, note when the project must be delivered by.

## What are the non-goals for the project? What should we explicitly aim to not do?



### <ins>For the presentation:</ins>
- Motivation and why we need this: we will talk about the disconnect between causal inference and demand modeling and why it’s important to pay attention to it. 
- Outline of procedure and the value it brings. (both these steps are mostly from Brathwaite et al.)
- Demo for application on simulated dataset.
- Outline of deconfounder approach
- Demo of deconfounder approach using simulation
    - Sensitivity to number of deconfounder
    - Sensitivity to different DAG assumptions. 
- (Potentially) Application of deconfounder approach to real dataset. 
- Conclude

### <ins>Actual work/code:<ins>
1. For simulation:
    - Function to simulate data from structural models (mainly linear models -- MNL and regression). 
        - Input: X variables, coefficients. 
    - Output: Simulated variables 
2. For deconfounder
    - Function for fitting factor model
        - A class with a few of factor model methods: Probabilistic PCA, Deep exponential family, Poisson Matrix Factorization?
           1. Input: Vector of covariates, dimensionality of latent variable space
           2. Output: A fitted factor model
        - Function for posterior predictive checks for the factor models.  
           1. Input: Factor model, test data
           2. Output: P-values for predictive checks
