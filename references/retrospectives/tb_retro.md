Retrospectives
==============
This document should store one's weekly reflections on the project. In particular, note

- one thing you learned this week while working on the project
- one thing that went well with the project this week
- one thing that went poorly with the project this week
- one work-process suggestion to improve the project next week


1. Week of February 17th, 2020
   1. I learned (or maybe finally appreciated) that the notion of modeling one's unobserved /  latent confounders is a much more general and older idea than the implementation shown by Wang and Blei in their deconfounder work.
   2. That Amine and Hassan both have a first draft of the structure for their simulations is great.
   3. I didn't schedule our weekly call in advance and coordinate to make sure everyone was available.
   4. Set up google calendar invite for weekly meeting.

2. Week of February 24th, 2020
   1. I created a mental schema of the various approaches to falsifying one's causal graph, and I learned that causal graphs can be used to encode assumptions of various experimental designs / identification strategies such as propensity score matching, regression discontinuity, etc. This led me to think that we should use causal diagrams to depict model-based identification strategies and pictures of how the world works, e.g. showing the computational graph of our model as the subgraph between our outcome variable Y and its parents.
   2. I read a lot of new, informative papers on checking one's causal graph, graphical models of preferences and utility in computer science, and causal inference + reinforcement learning.
   3. I didn't carefully track my time while working on the project so I don't actually know how much time was spent on it (though I have good lower bounds).
   4. We should have a draft work plan by the end of next week that takes us all the way to the conference itself.

 3. Week of March 1st, 2020
    1. I learned about tons of ways neural network researchers perform "prior predictive checks" of various graph structures, all under names that are not "prior predictive checks". Lots of work on network pruning, random weight assessments in neural architecture search, etc.
    2. Amine and Hassan made great progress on the project architecture and project workflow!
    3. I didn't ensure that I made time to test my implementation of the base model class.
    4. In our weekly meeting, we should decide which of our tasks we want to accomplish by midweek to make sure peer-reviews of each others work can happen! And we should schedule in time to review each others work.

4. Week of March 8th, 2020
   1. I learned about the notion of t-separation and how it can be used to identify "vanishing tetrads," aka how it can be ussed to derive testable implication from causal graphs with latent variables.
   2. I think the mid-week deliverables worked well to allow time for review of each other's work and helped ensure the completion of risky tasks (i.e. tasks that could easily take longer than anticipated).
   3. We are still lacking an exact plan that we expect to be doable within 3 weeks (Mar 15 and onwards) to produce the raw results to be used in the presentation for June.
   4. On pull-request reviews, specify at least 1 concrete question that you want your reviewees to answer.

5. Week of March 15th, 2020
   1. I learned about triad constraints and how they represent another class of, seemingly useful, testable implications of causal graphs with latent variables.
   2. Amine was able to create the utility level causal graphs for all utilities!
   3. I did not thoroughly and explicitly justify my belief in why I expected changing the causal structure of our explanatory variables would impact our ability to recover **unbiased** estimates of our model parameters. More specifically, I conflated the inability to recover an unbiased estimate of a given causal effect with the inability to recover an unbiased estimate of generative model parameter. Choice modeller's already know about this though. See the CE264 hw on forecasting.
   4. Spend 5 minutes reviewing one of the project planning documents each week. For example, the problem statement, vision, requirements etc. Will help us keep the big picture in mind.

6. Week of March 22nd, 2020
   1. I learned that prior and posterior predictive checks that depend on latent variables (i.e. so called discrepancy variables) can be used to test assumptions about the structure of one's causal graph. The idea is to use marginal or conditional independence tests on the simulated data + latent variabble vs the observed data + latent variable, integrated over the latent variable.
   2. Hassan showed that when using the mean of an inferred distribution of one's deconfounder in one's outcome model, the deconfounder has to be recovered almost exactly in order to get outcome model parameter estimates that are close to one's data generating parameters. It's great to learn the conditions under which we expect the deconfounder technique to work.
   3. We ran into yet more difficulties getting the deconfounder to work as expected.
   4. We should revise the project requirements since we are unlikely to actually finish a successful demonstration of the deconfounder approach by our internal deadline (April 5th, 2020).

7. Week of March 29th, 2020
   1. Somehow I might have accidentally failed to do a retro this week.

8. Week of April 5th, 2020
   1. I learned that building causal graphs with latent variables is likely to involve a ton of work that is not commonly spoken of. For instance, if one wants a causal graph with latent variables taht a-priori seems worthwhile investigating, then one will likely have to do a lot of work to create a generative model based on those latent variables that actually generates data that is similar to one's observed data. Standard factor models produce data that is far from realistic.
   2. Selection-on-observables simulation is prototype complete and so are the basic demonstrations of falsification checks for one's causal graph. Deconfounder investigation is almost complete as well. All in all, we appear on track for a June 1st completion of the presentation and planned work!
   3. The daily check-in's did not work. Four out of the 7 days since our last meeting, I was the only update on the issue.
   4. We should complete and revise/update the project planning documents (vision, requirements, architecture, project plan).
