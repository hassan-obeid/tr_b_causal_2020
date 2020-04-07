---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# <ins>Vision</ins>


## At the highest level, how to solve this project’s problem---what will be done in the project?

We will demonstrate, using simulations and a case study, how to approach travel demand modeling while paying attention to both the causal structure of how people make their choices and the treatment assignment process.
To do this, we will apply the causal inference workflow outlined in Brathwaite and Walker (2018), and we will augment the workflow with additional details and guidance from our experience's in this project.
We will also demonstrate the value of such an approach by contrasting the outcome of this way of thinking to that of traditional demand modeling where little care is given to the underlying causal structure of the problem.

Specifically, there will be three main deliverables in our work/presentation.

The first deliverable is a simple application of the steps outlined in Brathwaite and Walker (2018) to a simulation example.
Here, we will show what would happen to our causal estimates under different assumptions of the causal graph.
In this example, the problem will be purely a selection-on-observables problem, and there won’t be complications related to unobserved latent confounders.
The goals of the simulation will be to (1) demonstrate the step-by-step causal inference workflow, and (2) show the pitfalls of ignoring the treatment assignment mechanism.

The second deliverable of our work will be a combined simulation study and case study.
This effort will focus on the more complicated and realistic case that is typically faced by demand modelers, where we have latent confounders that affect the treatment assignment of our causal variables of interest.
Taking a high-profile and recent approach to coping with latent confounding, we will apply the de-confounder technique of Wang and Blei (2019).
Our application will show how directed acyclic graphs can help increase transparency about one's reasoning regarding the number of confounders, assumptions for which variables are confounded, and the models needed to estimate the causal effects.
We will investigate, through simulation, the usefulness, and pitfalls/sensitivity of this approach, and we will demonstrate the technique in a travel demand  modelling applicaton with real world data.

The final deliverable of our work will be an empirical demonstration of methods for falsifying one's causal graph.
While such efforts are implicit in each of the deliverables above, we will draw special attention to them here.
Our rationale is that modelers are likely to (1) skip this step without clear guidance and (2) begin statistical estimation before assumption falsification.
Here, we will present simple techniques that modelers can use to avoid the catastrophic delusion and incorrect inferences that comes from using obviously incorrect causal graphs (e.g. see the first deliverable).
