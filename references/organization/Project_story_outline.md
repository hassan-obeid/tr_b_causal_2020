---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.3.4
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# <ins>Story</ins>



### Part 1: Selection on observables

We show that an outcome model, without a data generation model, is not enough when it comes to making causal inferences. Inferences from an outcome model alone assume a very specific data-generation process (namely, one where all the covariates are independent), and it is important to make that clear for transparency. 

To do that, we simulate a mode-choice dataset from a known outcome (MNL) model as well as an assumed causal graph. We then re-estimate a logit model on the simulated dataset, and try to estimate the causal effect of perturbing travel distance on the mode share of auto, using two ways:

 - The first is when the change in travel distance doesn't affect any of the outcome variables.
 - The second is when the change in travel distance leads to a change in its descendants (namely like travel cost and time). Here, we're assuming that we somehow were able to know the causal graph as well as the parametric relationships between its nodes. In reality however, one needs to recover the causal graph by iterating through reasonable alternatives and attempting to falsify them using the conditional independence tests outlined [here: reference to Timothy cit notebooks]. Once a causal graph is obtained, the modeler has to then model the dependence between the variables using any parametric or non-parametric specifications that she deems appropriate, of course using appropriate tests of statistical goodness-of-fit. 
 
We run the process above for N simulations and obtain a distribution of the causal effect of interest. The purpose of this exercise is to show that talking about causal effects without explicitely having a model of the data generation process leads to significant bias in the results. 

Possible extensions:
So what we've done now is show that a causal effect estimated "naively" is biased, but we haven't shown how to construct the causal graph, or estimate the relationships between the nodes, which the modeler has to do. I'm thinking we can present the problem using three levels of hierarchy for contributions:
 - We don't know the causal graph. 
 - We know the causal graph, but we don't know the equations on the edges. 
 - We know the causal graph, and we also know the equations on the edges. 

Our work so far has focused on level 3. Is there value of a demonstration for level two? Or should we just outline the steps?

### Part 2: Selection on observables

The details for this is basically outlined in the "investigating deconfounder" memo. The story we're telling can be framed around what the challenges are with latent confounding, and how to deal with it when you only have access to observational data and can't conduct surveys/indicator questions. We then talk about one specific way of dealing with latent confounding, using the recently proposed deconfounder method by Blei et al. 

We then go into the details of the deconfounder method (model formulation, steps needed to perform it, required tests, etc. -- details from Blei et al.). Since it's a recent method, not a lot has been done to assess its usefuleness in practice. We do that, and run a simple simulation exercise where we simulate data with latent confounding, and attempt to recover the confounder using the deconfounder method, and find that even with the slightest inaccuracies in the recovered confounder, the re-estimated parameters controlling for the confounder still suffer from significant biases. 

```python

```
