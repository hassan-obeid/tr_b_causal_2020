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
    1. Started learning about testing and writing proper classes
    2. Produced full class for function to simulate data independently based on distributions 
     and full functions for simulating choice data
    3. Haven't been able to write solid tests/finish the architecture doc
    4. Set up some intermediate deadline for each of the weekly tasks to be more organized with the workflow

3. Week of March 2nd, 2020
    1. Started learning a lot more about the internal functionality of pylogit and where everything is located
    2. Have been able to finish a full draft of architecture document and full draft
    3. I found that I sometimes didn't have a wide enough vision of what the different potential problems I can run into while writing code. I underestimated that I can run into bugs that might not be easy to fix. This relates to estimates of time of each task as well.
    4. Based on tim's suggestion, I found it very very useful to commit/push as often as possible so that the deltas are more meaningful and easier to track. So I would say we should strive to do that very often whether it is with code, documentation, tests, or anything related to the project really.

4. Week of March 9th, 2020
    1. Learned a little about distributional regression and more pythonic ways of writing code in general
    2. Full simulation code, responded to comments on the architecture document and produced a workflow for the simulation based on the causal graph
    3. I failed at writing code to remove outliers from the simulated data, even if I think this might not be necessary anymore?
    4. I think we would have to move potential deadlines for important issues earlier in the week

5. Week of March 15th, 2020
    1. Learned about how we can set up our experimentation protocol within sacred and some of the simulation issues we can run into (e.g.:joint distribution simulations)
    2. Finished writing code to simulate data from causal graph based on workflow
    3. I failed at writing code to remove outliers from the simulated data, I also failed at following the exact workflow due to some challenges with storing data and the lack of flexibility within `causalgraphicalmodels`
    4. Maybe daily checkins? even one sentence on the repo saying what you worked on that day, even if the work didn't result in any commits.

6. Week of March 22nd, 2020
    1. Learned a little more on how to write better functions. More learning to come on how causal discovery works
    2. Finished the task list and the simulation efforts
    3. I wasn't able to all cases of logistic regression in our regression.
    4. Continue our daily checkins. I think they've been useful.
    
7. Week of March 29th, 2020
    1. Running simulations take a much longer time than expected. I think we need to modify our functions to parallelize this process.
    2. Finished the causal effect estimation workflow.
    3. I didn't post daily checkins on a daily basis. 
    4. I think it's time we start actively thinking about the story we want to tell at the conference and start refactoring what needs to be refactored.
    
8. Week of April 5th, 2020
    1. I learned how much I don't know is within pylogit, and how much I don't overall know about refactoring, refactoring to me was just writing better variable names with clear operations.
    2. I started reading the refactoring guru guide, even if at times I found myself just staring at it because I didn't understand some of the refactoring techniques or their purpose.
    3. I didn't update the architecture document.
    4. I think we should probably start thinking of what the direction after the project should be, like how this project could lead to research work for our phd research/dissertations?

9. Week of April 5th, 2020
    1. Some more learning about how refactoring is supposed to be done/how much time it takes.
    2. I finished rewriting all relevant functions to be refactored. Pending test writing.
    3. Did not finish the presentation.
    4. I think it's time to bring in Joan/Vij this week for discussion/suggestion.
