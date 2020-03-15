Retrospectives
==============
This document should store one's weekly reflections on the project. In particular, note

- one thing you learned this week while working on the project
- one thing that went well with the project this week
- one thing that went poorly with the project this week
- one work-process suggestion to improve the project next week

1. Week of February 17th, 2020
  1.
  2.
  3.
  4.


2. Week of February 24th, 2020
    1. Learned about posterior model checking. 
    2. Created a posterior checking procedure with p-values and individual graphs. 
    3. Underestimated the time it takes to learn concepts related to bayesian model checking, and the time it would take to code them from scratch. Adapting the code from the deconfounder is not as straightforward either. 
    4. I think we need to be more focused on the exact objectives for this specific conference deliverable. The objective is not to show how to falsify causal graphs, but rather to demonstrate how to document analyses in causal graph and use those graphs for downstream analysis. 


3. Week of March 1st, 2020
    1. The importance of visualization before computing summary point-statistics
    2. Writing down a concrete project plan was a big relief and gave me clarity. Also, digging deeper into model checking revealed issues and complexities with how to fit and validate factor models that I was underestimating.
    3. I still don't completely understand the test statistic used for the posterior model checking in Blei's tutorial. I'm also noticing that PPCA might not be the best model to use to recover a latent confounder, and I need to learn about other potential models. 
    4. I think we're doing good progress in the right direction. I'm hoping we can nail down a procedure for validating factor models that aren't necessarily Bayesian (cross-validation scores?) -- I want to discuss this more. 
    
4. Week of March 8th, 2020
    1. Working from home is hard. 
    2. The mini-project of simulation generated good insights and unveiled some potential issues we will face. The conversation with Joan opened up some interesting prospects.
    3. Still can't get the deconfounder to make sense, even on a simple simulation. Including it in the final regression is not changing the coefficients on the parameters of interest. Maybe need to try more confounders.
    4. Nothing specific, we should just keep chugging along and making progress. 
