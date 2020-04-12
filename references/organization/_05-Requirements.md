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

# <ins>Requirements</ins>


## What exactly does the product need to deliver to meet the needs of the end users / stakeholders?

### <ins>For the presentation:</ins>
The presentation, per specification of the conference, can only be 20 minutes long.  
Accordingly, should have an expectation of 20 slides, with 15 -30 slides (min, max).

The structure of the presentation should be as follows:
- Motivation: Why is our problem important to the listener? (1 min)

- Need: What is the need being addressed? (1 min)  
Travel demand modelers wish and need to make causal inferences, but the field typically does not use techniques from the causal inference literature.

- Our Solution (1 min):
   - Provide a demonstration of using causal inference techniques in travel demand modeling context.
   - In particular, we demonstrate the causal inference workflow from Bratwaite and Walker (2018), with references to similar / equivalent ideas put forth by others, and with additional details from our experiences in this project.

- Main message of the talk (1 min):  
Specify a causal graph, make sure the assumptions of that graph are not violated by one's data, and build one's model on the basis of the causal graph.

- Point 1 (4 min):  
Without taking into account the treatment assignment mechanism / causal structure of one's explanatory variables, one's estimated treatment effects may be completely wrong.
   - Amine's selection-on-observables simulation results and description.

- Point 2 (6 min):  
When dealing with latent confounders in one' causal graph, one generically applicable technique is to model the latent confounders.
Pitfalls abound when applying such techniques, so we demonstrate / raise awareness of / and show how to detect such problems.
   - Hassan's deconfounder demo and simulation results.

- Point 3 (4 min):  
In order to use any of these techniques, having a well-specified causal graph is crucial. We demonstrate methods for checking / falsifying one's causal graph in order to avoid drawing erroneous or unsupported conclusions.
   - Conditional Independence Tests
   - Marginal Independence Tests
   - Prior and Posterior Predictive Conditional / Marginal Independence Tests

- Recap (1 min)

- Conclusion (1 min):  
What does this presentation mean for the audience?  
What can they now do that may have been mysterious / hard before?  
How do we want the audience's behavior change as a result of this presentation?

### <ins>Actual work/code:<ins>
1. Public notebooks:
   1. Selection-on-observables simulation:
      - Function to simulate data from structural models (mainly linear models -- MNL and regression).
        - Input: X variables, coefficients.
      - Output: Simulated variables
   2. Deconfounder demonstration with data from Brathwaite and Walker's asymmetric models paper.
   3. Deconfounder investigation / simulations
      - Function for fitting factor model
          - A class with a few of factor model methods: Probabilistic PCA, Deep exponential family, Poisson Matrix Factorization?
             1. Input: Vector of covariates, dimensionality of latent variable space
             2. Output: A fitted factor model
          - Function for posterior predictive checks for the factor models.  
             1. Input: Factor model, test data
             2. Output: P-values for predictive checks
   4. Demonstration of falsification techniques
2. Resulting plots / tables
3. Tests for all source code and notebooks.

## Do the end user’s / stakeholders have firm time deadlines for the project? If so, note when the project must be delivered by.

The original deadline for submitting the presentation materials for the conference was June 1st, 2020.  
The presentation and supporting code should be complete by then.

## What are non-goals for the project? What should we explicitly avoid?
1. Causal discovery.  
It is of the foremost importance to create a causal graph that describes the causal system, i.e. the treatment assignment process + outcome process that one is working with. However, we do not have time to do this well before June. Consequently, we will focus on simulations where we know the true causal graph, and we will provide a description and demonstration of the tools needed to carry out a complete causal discovery process. This will be done under point 3 of the presentation about falsifying one's causal graph.

2. Estimating functional relationships between EACH treatment variable and its parents.  
Corectly specifying the qualitative dependence relationships between each of the variables in one's system (i.e. drawing the correct causal graph) is only the beginning of the work needed to estimate one's desired causal effects. Next, one needs to estimate statistical models to describe the relationship between each treatment node and its parents and between the outcome nodes and their parents. As the estimation of statistical models is the most familiar aspect of causal inference problems, we will not dwell on it in this presentation. Instead our simulations will use the correct relationships between each treatment node and its parents for the sake of demonstration.

3. Generating causal effect estimates that we believe are correct.  
For our real data demonstration, our analysis will feature a number of known deficiencies.
   - We will be using a causal graph that we know is incorrect, based on various falsification tests.
   - We will be using functional relationships for treatment nodes given their parents that we know are incorrect based on posterior predictive checks.
   - We believe that we suffer from latent confounding in our causal system, but we do not believe that we have an adequate causal graph that represents this latent confounding, based on both prior and posterior predictive independence tests of our hypothetical causal graph.

   As a result, we do not expect nor hope to be making correct causal effect inferences.  
   We merely aim to demonstrate the process of how one would correctly estimate causal inferences in a travel demand setting.
