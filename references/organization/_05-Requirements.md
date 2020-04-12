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
Provide a demonstration of using causal inference techniques in travel demand modeling context.
   - In particular, we demonstrate the causal inference workflow from Bratwaite and Walker (2018a), with references to similar / equivalent ideas put forth by others, and with additional details from our experiences in this project.
   - We use data from Brathwaite and Walker (2018b), a study of traveler mode choice in the San Francisco Bay Area.
   The setting is one related to large tech companies considering how they could reduce the driving mode-share of their employees by moving them closer to work.
- Main message of the talk (1 min):  
Specify a causal graph, make sure the assumptions of that graph are not violated by one's data, and build one's model on the basis of the causal graph.

- Point 1 (4 min):  
Without taking into account the treatment assignment mechanism / causal structure of one's explanatory variables, one's estimated treatment effects may be completely wrong.
   - Amine's selection-on-observables simulation results and description.

- Point 2 (6 min):  
When dealing with latent confounders in one' causal graph, one generically applicable technique is to model the latent confounders.
Such techniques are may substantially change one's results.
However, pitfalls abound when applying these methods, so we demonstrate / raise awareness of / and show how to detect such problems.
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
   - We want them to check out our paper and github repo to see additional details.  
   The presentation is an advertisement not a sufficient / stand-alone teaching tool.
   - We want them to use (and be able to use) our proposed causal inference workflow.

### <ins>Actual work/code:<ins>
1. Public notebooks:
   1. Selection-on-observables simulation:
      - Function to simulate data from structural models (mainly linear models -- MNL and regression).
        - Input: X variables, coefficients.
      - Output: Simulated variables
   2. Deconfounder demonstration with data from Brathwaite and Walker's asymmetric models paper.
      - Does the deconfounder approach substantially change the estimated causal effects and inferred sensititivities?
   3. Deconfounder investigation based on simplified simulations
      - Simplified simulations to illutrate potential pitfalls of the deconfounder approach.
   4. Deconfounder demonstration using realistically simulated data based on data in Brathwaite and Walkders asymmetric models paper.
      - Realistic simulation to confirm whether the results obtained with the real data are qualitatively consistent with results obtained on data that we know satisfy the deconfounders assumptions.
   4. Demonstration of falsification techniques.
2. Resulting plots / tables
   - Selection-on-observables simulation
      - Correctly estimated vs True causal effect of travel distance reduction on automobile mode shares (drive alone + shared_ride_2 + shared_ride_3+).
      - Naively estimated vs True causal effect of travel distance reduction on automobile mode shares (drive alone + shared_ride_2 + shared_ride_3+).
   - Simplified / Illustrative Deconfounder simulations
   - Deconfounder demonstration with real data
      - Plots of asymptotic distributions of model coefficients with and without the inferred deconfounders.
      - Plots of predicted distributions (based on the asymptotic distribution of model coefficients) of causal effects with and without the inferred deconfounders.
   - Deconfounder demonstration with realistically simulated data.
      - Plots of asymptotic distributions of model coefficients with and without the inferred deconfounders.
      - Plots of predicted distributions (based on the asymptotic distribution of model coefficients) of causal effects with and without the inferred deconfounders.
      - Comparison of the two plots above next to those same plots based on the real data.  
      We want to know if the results observed using the real data are qualitatively consistent with the results obtained using data that we know satisfies the deconfounder assumptions.
   - Falsification tests
      - Causal graph for Utility Drive Alone.
      - Marginal independence test statistic distribution vs observed value, for causal graph of Utility Drive Alone graph.
      - Conditional independence test statistic distribution vs observed value, for causal graph of Utility Drive Alone graph.
      - Deconfounder causal graph for Utility Drive Alone
      - Prior and posterior predictive test statistic distribution for conditional independence test statistics vs observed test statistic value.
3. Supporting source code.
   - Selection-on-observables
      - Function(s) for estimating some statistical model for each treatment node given its parents.
      - Function for simulating data from a specified causal graph, the estimated statistical models of each treatment node given its parents, and a given outcome model given the treatment nodes.
      - Function for re-estimating a given outcome model, conditional on a set of simulated data for the parents of the outcome nodes.
      - Function for estimating causal effects given a specified causal graph and relationships between  nodes and their parents.
      - Function for plotting the distributions of estimated causal effects under various causal graphs and relationships between nodes and their parents.

   - Deconfounder
      - Function for fitting factor model  
        A class with a few of factor model methods:  
        Probabilistic PCA, Deep exponential family, Poisson Matrix Factorization?
          1. Inputs:
             - Matrix of covariates
             - Dimensionality of latent variable space
             - Type of factor model to be estimated
          2. Output: A fitted factor model
      - Function for prior predictive checks of the factor models.
         1. Inputs:
            - Factor model
            - Training data
            - Prior distributions of factor model parameters
         2. Outputs: P-values and plots for predictive checks
      - Function for posterior predictive checks for the factor models.  
         1. Inputs:
            - Factor model
            - Training and/or testing data
            - Posterior distributions of factor model parameters
          2. Output: P-values and plots for predictive checks
   - Falsification of causal graphs
      - Function for marginal independence tests
      - Function for conditional independence tests
      - Function for prior/posterior predictive conditional independence tests
4. Tests for all source code and notebooks.
   1. [At Minimum] Integration tests of public notebooks.
   2. [At Minimum] Unit tests of all critical / top-level / non-trivial / non-standard functions.
   3. [Ideally] Unit tests of all source code.

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
