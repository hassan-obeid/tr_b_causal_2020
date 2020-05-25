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
   1. I created a mental schema of the various approaches to falsifying one's causal graph, and I learned that causal graphs can be used to encode assumptions of various experimental designs / identification strategies such as propensity score matching, regression discontinuity, etc.
   This led me to think that we should use causal diagrams to depict model-based identification strategies and pictures of how the world works, e.g. showing the computational graph of our model as the subgraph between our outcome variable Y and its parents.
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
   3. I did not thoroughly and explicitly justify my belief in why I expected changing the causal structure of our explanatory variables would impact our ability to recover **unbiased** estimates of our model parameters.
   More specifically, I conflated the inability to recover an unbiased estimate of a given causal effect with the inability to recover an unbiased estimate of generative model parameter.
   Choice modeller's already know about this though. See the CE264 hw on forecasting.
   4. Spend 5 minutes reviewing one of the project planning documents each week. For example, the problem statement, vision, requirements etc. Will help us keep the big picture in mind.

6. Week of March 22nd, 2020
   1. I learned that prior and posterior predictive checks that depend on latent variables (i.e. so called discrepancy variables) can be used to test assumptions about the structure of one's causal graph.
   The idea is to use marginal or conditional independence tests on the simulated data + latent variable vs the observed data + latent variable, integrated over the latent variable.
   2. Hassan showed that when using the mean of an inferred distribution of one's deconfounder in one's outcome model, the deconfounder has to be recovered almost exactly in order to get outcome model parameter estimates that are close to one's data generating parameters.
   It's great to learn the conditions under which we expect the deconfounder technique to work.
   3. We ran into yet more difficulties getting the deconfounder to work as expected.
   4. We should revise the project requirements since we are unlikely to actually finish a successful demonstration of the deconfounder approach by our internal deadline (April 5th, 2020).

7. Week of March 29th, 2020
   1. Somehow I might have accidentally failed to do a retro this week.

8. Week of April 5th, 2020
   1. I learned that building causal graphs with latent variables is likely to involve a ton of work that is not commonly spoken of.
   For instance, if one wants a causal graph with latent variables that a-priori seems worthwhile investigating, then one will likely have to do a lot of work to create a generative model based on those latent variables that actually generates data that is similar to one's observed data.
   Standard factor models produce data that is far from realistic.
   2. Selection-on-observables simulation is prototype complete and so are the basic demonstrations of falsification checks for one's causal graph.
   Deconfounder investigation is almost complete as well.
   All in all, we appear on track for a June 1st completion of the presentation and planned work!
   3. The daily check-in's did not work. Four out of the 7 days since our last meeting, I was the only update on the issue.
   4. We should complete and revise/update the project planning documents (vision, requirements, architecture, project plan).

9. Week of April 12th, 2020
   1. I learned / realized that I have very little understanding of our presentation audience. I have not read or cannot easily summarize the causal inference related papers or thoughts of most conference attendees.
   2. Revising the vision and requirements doc was very useful for me to firmly center the big picture of the project in my mind and remind me that we've completed most of the essential / minimally-viable work.
   3. I failed to complete my assigned refactoring and review tasks for the week. This revealed that I made my plans without adequate respect/consideration for the additional activities that were bound to happen this week (prepping for the lecture in CE264). Moreover, I planned without any slack in the schedule so I also could not respond to the unexpected life-related tasks (taxes, coronavirus news and communication with loved-ones, miscellaneous necessary duties) that popped up and took much of my time.
   4. We should probably complete one / two last piece of work for the project so travel demand modelers find it practically relevant.
      - We should explicitly show the change in outcome model parameters and causal effects with our real dataset from using the deconfounder vs not. This will show whether the techniques are practically relevant.
      - We should show whether our deconfounder results with real data vis-a-vis non-deconfounder results with real data are qualitatively consistent with our deconfounder results with realistically simulated data vis-a-vis non-deconfounder results with realistically simulated data. This will give us evidence about the trustworthiness of the qualitative conclusions of our results.

10. Week of April 19th, 2020
    1. Life was crazy, and I never updated the retrospective.

11. Week of April 26th, 2020
    1. Life was crazy, and I never updated the retrospective.

12. Week of May 3rd, 2020
    1. Life was crazy, and I never updated the retrospective.

13. Week of May 10th, 2020
    1. This week, I learned more about myself. Specifically, I learned what I believe our contributions in this project are:
       1. clearly demonstrating, in the context of discrete choice models, that one should care about the causal mechanism generating your explanatory variables.
       The point being demonstrated is not new, but I believe the clear demonstration itself is a contribution.
       2. demonstrating that drawing credible causal inferences in the face of unobserved confounding is hard AND that recent techniques for doing so (i.e. the deconfounder) do not immediately work well.
       I do not believe this point about the deconfounder not immediately working well (for the reasons you found) has been made anywhere else; I therefore think it is a contribution.
       3. demonstrating (again in the context of discrete choice) that there are fast, easy to use tests that show when one’s causal assumptions are grossly violated.
       I believe this information can be critically useful, yet it is not widely known or written about in the discrete choice literature.
    2. Talking with Joan and Vij last week was great for bringing up points where traditional choice modellers may (i) take issue with our work and (ii) see value in our work:
       - thinking that the emphasis on the generating model for explanatory variables is old news due to the existence of activity based models.
       - thinking that the methods for dealing with unobserved confounding are "just" hybrid choice models where the indicators are themselves explanatory variables
       - being excited about ways to test different causal structures / orderings in an activity based model (e.g. destination choice before mode choice or vice versa).
    3. It was again the case that I accomplished less than I wanted to in the week.
    Job hunting, interview prepping, and doing research on the side is harder than  I imagined.
    4. This week, I think I should only attempt to do one thing: refactor my code.
    It is unlikely that I can review the code of others and edit my own code, given the other job-search-related demands on my time.

14. Week of May 17th, 2020
   1. Retrospective was never updated and weekly meeting was altered due to UC Berkeley finals.

15. Week of May 24th, 2020
   1. Learned that one needs to perform model checking when doing model-based testing of conditional mean independence (i.e., an implication of conditional independence).
   One needs to extract all predictive power from the conditioning variables when testing whether or not a given variable X2 is independent of X1, conditional on a set of variables Z.

   2. Nothing with the project this week went better than expected.
   3. I didn't finish refactoring the notebook for testing of latent, conditional independence assumptions.
   4. For the upcoming week I should focus on even less: just refactoring that one notebook on latent, conditional independence assumptions.
