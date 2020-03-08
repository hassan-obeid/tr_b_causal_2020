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
