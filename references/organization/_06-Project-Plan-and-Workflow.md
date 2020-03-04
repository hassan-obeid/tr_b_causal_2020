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

# <ins>Project Plan and Workflow</ins>


## What is the plan for this project and the high level workflow? <br>

### Causal graphs are fundamental to any model where parameters are assigned causal interpretations, even under selection on observables.
The goal here is to show that even for the same outcome model (e.g. mode choice model), the model’s ability to recover the true data generating parameters depends heavily on the data generating process. The idea here is to demonstrate that when we control for intermediate variables of some variable of interest, we never recover the true causal parameter on the variable of interest asymptotically. <br>
We demonstrate this by simulation, where the work is split into two parts, both working off of the asymmetric paper data by Brathwaite (2018): <br>
-	In the first part, we simulate data from a causal graph where all explanatory variables are independent and observed (there are no structural relationships between the variables). We then simulate an outcome mode choice using a well-defined MNL model. This will be our population data. We then sample from this population multiple times and re-estimate the outcome model, and look at the percentage of time that we recover the true parameters on our variables of interest. We should find that we recover the parameters almost all the time.  <br>
-	In the second part, we use the same variables, but now we simulate the data based on a more realistic causal graph where some variables influence/cause the others. We then simulate choice data based on the same model used in part 1. Finally, we sample again from this population multiple times and re-estimate the outcome model, and look at the percentage of time that we recover the true parameters on our variables of interest. What we should find here is that asymptotically, we never recover the true parameters on any variables that have descendants in the causal graph that we control for. <br>
The conclusion from this is that even though the outcome model is the same for the data generation process, we can only assign causal interpretations to parameters if we have clearly drawn our assumptions on the data generating process. Without those, one may easily challenge any interpretation we assign to our parameters by simply asking the question “but what if the data generating process looked like this?”. By drawing a causal graph upfront, we limit this sort of questions to questions about the validity of one’s causal graph, which comes with testable implications and create much more interesting and focused discussions than those created when there’s no causal graph that clearly shows the modeler’s assumptions. 

##### Concrete steps:

Functions <br>
-	The main function we need is one that takes as input a hypothesized causal graph (through a set of nodes and edges), with distributional assumptions and parametric relationships between variables, and simulates data given those inputs. 
-	Since we’re dealing with a multinomial choice problem, we need a function to simulate outcomes and be flexible to take in multiple specifications for each utility equation.

Workflow <br>
-	Come up with an outcome (choice) model based on the bike data in Brathwaite (2018). The model doesn’t need to be the exact model estimated in the paper, but that can be the case. We just need to select a set of variables while being mindful that one of those variables will be used as a latent confounder in the next step. 
-	Simulate two sets of data for the X variables: <br>
    - One where all the Xs are independent (i.e. the only edges in the causal graph are between the X and the outcome variable). 
    - One based on a realistic causal graph with confounders. Preferably focus on a case with one latent confounder (like distance? Age? Gender?) <br>
-	For each of the two causal graphs above, simulate choice data based on the outcome model assumed. <br>
-	Estimate the choice model for each of those two datasets. <br>
-	Repeat the process n times, and calculate the percentage of those times where the estimated parameters’ confidence intervals contain the true parameter. <br>

### Dealing with latent confounders in the causal graph

In this part of the work, we take the realistic causal graph from part 1 and remove a latent confounder from it. We then attempt to recover the latent confounder using the de-confounder algorithm proposed by Blei et al. (2018). We will apply this work to the simulated data set, where we know the true parameters of interest. We will repeat the same simulation process in part 1 where we sample from the population and estimate models where we control for the latent confounder using the deconfounder approach and look at the percent of time we recover the true parameter. 

##### Concrete steps: 

Functions <br>
-	A class with functions to estimate factor models. Start with PPCA as in the deconfounder’s tutorial and expand as needed. <br>
-	A function to perform posterior predictive checks on the factor model to assess its predictive accuracy. <br>

Workflow <br>
-	Start with a hypothesized general causal graph. For the simulated example, we’ll start by assuming the true causal graph used to generate the data, where we highlight the confounder as being latent/unobserved. <br>
-	Then, come up with a “utility level” causal graph, where we come up with a causal graph for each utility equation since the confounding is at the level of the utilities in a MNL context. These causal graphs would inform the utility specifications, and will determine which utilities do and do not require estimating a latent confounder. We discussed this briefly before, but are we on the same page? <br>
-	Based on the above specifications, estimate the required latent confounders using a factor model, start with PPCA. <br>
-	Perform posterior predictive checks on the estimated factor model and assess goodness of fit. Since we’re doing this on a simulated dataset, then we know in advance how many latent confounders there are. We could potentially try controlling for more confounders and see how that affects the results (in other words, see how sensitive our analysis is to the number of confounders used). <br>
-	Once the checks pass, include the latent confounder in the utility equations, and estimate a MNL. <br>
-	Repeat the process n times, and calculate the percentage of those times where the estimated parameters’ confidence intervals contain the true parameter. <br>
Finally, we can repeat the above process, but now on the full, real bike dataset. The workflow would be the same as above, with the addition of one step: <br>
-	After hypothesizing on a causal graph, we need to do some conditional independence testing to falsify the graph by making sure nothing clearly violates conditional independence tests. The caveat here is that there don’t seem to be one way to perform conditional independence testing – there are many ways proposed, but I’m not sure if the problem itself is solved yet. We can try one way, and point out that this is an active area of research and that there exists other ways to do it. The important take away is that the causal graph should not clearly violate a given test, in which case something is obviously wrong. <br>
-	Once a non-falsifiable causal graph is nailed down, we proceed exactly as above. 


