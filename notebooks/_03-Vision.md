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

# <ins>Vision</ins>


## At the highest level, how to solve this project’s problem---what will be done in the project? 

We will demonstrate using simulations how to structurally approach demand modeling and modeling in general while paying attention to the causal structure of how people make their choices, and the data generation process in general. 
We will build on Brathwaite’s work and review paper and demonstrate with steps how to incorporate and think about causal inference in the context of DCM. We will also demonstrate the value of such an approach by contrasting the outcome of this way of thinking to that of traditional demand modeling where little care is given to the causal structure. 

Specifically, there will be two main deliverables in our work/presentation. The first one is a simple application of the steps outlined in Brathwaite et. al to a simulation example, where we will also show what would happen to our causal estimates under different assumptions of the causal graph. In this example, the problem would be purely a selection on observables problem, and there won’t be complications related to unobserved latent confounders -- the goal is to solely demonstrate the step-by-step approach with explicitly stating and showing the causal assumptions being made. 

The second part of the paper will focus on the more complicated and realistic case that is typically faced by demand modelers, where we have latent confounders that affect the selection process into our causal variables of interest. We will apply the de-confounder approach by Blei et al. and demonstrate how DAGs can help reason/increase transparency about the number of confounders, assumptions for which variables are confounded, and models needed to estimate the causal effects. We will also demonstrate this approach using simulations, and quantify the sensitivity of this approach under different assumptions of variables confounded, number of confounders, and DAGs. (Maybe) apply this approach to a real world dataset as well. 
