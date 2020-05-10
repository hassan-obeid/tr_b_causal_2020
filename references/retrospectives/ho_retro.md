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
    1. Learned about ICLVs and how they compare to our work. Working from home is hard. 
    2. The mini-project of simulation generated good insights and unveiled some potential issues we will face. The conversation with Joan opened up some interesting prospects.
    3. Still can't get the deconfounder to make sense, even on a simple simulation. Including it in the final regression is not changing the coefficients on the parameters of interest. Maybe need to try more confounders.
    4. Nothing specific, we should just keep chugging along and making progress. 


5. Week of March 22, 2020
    1. Learning about mixed regressions and regressing on a distribution at the individual level. I have more to learn/ask about here. 
    2. Gained a better understanding of the sensitivity of causal estimates to small changes in the confounder. 
    3. Unclear how to make the deconfounder algorithm useful when we're bound to recover it with some degree of error. 
    4. Nothing specific, we should just keep chugging along and making progress. 
    
6. Week of March 29, 2020
    1. 
    
    2. Worked with Amine on nailing down the workflow for estimating the causal effects and proving the importance of having a data generation model (model of Xs) on top of an outcome model. I also brainstormed potential ideas to for our story and to make the work more relevant with steps that modelers can follow when looking to estimate causal effects. 
    
    3. Nothing specifically went wrong this week. 
    
    4. I'll start working on putting together a presentation that follows the flow of our story. I'll be presenting this to Joan's group in two weeks. 
    
7. Week of April 5, 2020
    1. Learned extensively about causal processes in time series analysis. 
    
    2. Thinking about a useful story that explicitely states our contributions, and coming up with a coherent structure for our presentation/the way we want to show our work in general. 
    
    3. Didn't get to finish my task list, specifically the refactoring I was planning on doing. 
    
    4. I'll start working on putting together a presentation that follows the flow of our story. I'll be presenting this to Joan's group in two weeks. 
    
7. Week of April 12, 2020
    1. Started reading about work done on the limits of the deconfounder algorithm. I have three papers to read on "the challenges of multiple causes", and will digest what applies to travel demand modeling when I read them. 
    
    2. Finishing up the presentation that has a clear story and key takeaways. Started working on refactoring.
    
    3. Nothing specific. 
    
    4. Will investigate setups where the deconfounder can work. It must work under some strict conditions and simulation scenarios, and I want to figure out what those are. 
    
8. Week of May 3rd, 2020
    1. Learned more about collider bias in an attempt to think about how to find an example where not paying attention to causality can lead to serious mistakes and bias.
    
    2. The bigger group conversation was useful and we got some good feedback on how to more clearly define and communicate our contribution to the field. 
    
    3. Nothing specific. 
    
    4. Wrapping up the first stage of the work (the work we've done so far) and have it in a format that is ready for public viewing. 
