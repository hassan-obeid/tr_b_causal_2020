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
Overall, our efforts are meant to
1. show the need to consider one's treatment assigment mechanism when estimating causal effects
2. highlight pertinent considerations when dealing with unobserved confounding, and
3. promote simple methods of testing/falsify one's causal graph (i.e. one's assumed treatment assigmnent mechanisms).

Corresponding to each of these goals, there will be three main deliverables in our work/presentation. Each of these deliverables will be based either on a real world dataset of travel mode choice decisions in the San Francisco Bay Area or on simulated datasets that mimic the real one.

The first deliverable is a simple application of the steps outlined in Brathwaite and Walker (2018) to our simulated datasets.
The main quantitative result of these simulations will be a comparison of the effictiveness of our proposed causal inference workflow for estimating average causal effects versus approaches followed in traditional demand modeling.
In this example, the problem will be purely a selection-on-observables problem, and there won’t be complications related to unobserved latent confounders.
The goals of the simulation will be to (1) demonstrate the step-by-step causal inference workflow, and (2) show the pitfalls of ignoring the treatment assignment mechanism as is traditionally done.

The second deliverable of our work will be a combined simulation study and case study.
This effort will focus on the more complicated and realistic case that is typically faced by demand modelers, where we have latent confounders that affect the treatment assignment of our causal variables of interest.
Taking a high-profile and recent approach to coping with latent confounding, we will apply the de-confounder technique of Wang and Blei (2019) to our real world dataset.
Our application will show how directed acyclic graphs can help increase transparency about one's reasoning regarding the number of confounders, assumptions for which variables are confounded, and the models needed to estimate the causal effects.
In addition to the application above, we use simplified simulation scenarios to further investigate the usefulness and pitfalls/sensitivity of this approach for generating accurate model estimates.

The final deliverable of our work will demonstrate methods for falsifying one's causal graph using the case study / real world data mentioned above.
While such efforts are implicit in each of the deliverables above, we will draw special attention to them here.
Our rationale is that modelers are likely to (1) skip this step without clear guidance and (2) begin statistical estimation before assumption falsification.
Here, we will present simple techniques that modelers can use to avoid the incorrect inferences that come from using obviously incorrect causal graphs (e.g. see the first deliverable).
