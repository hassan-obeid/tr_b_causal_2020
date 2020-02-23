# Expected Debugging Protocol [Work-In-Progress]

Based loosely on material from:
- Zeller, Andreas. Why programs fail: a guide to systematic debugging. Elsevier, 2009.
- Agans, David J. Debugging: The 9 indispensable rules for finding even the most elusive software and hardware problems. Amacom, 2002.

All suggestions for additional sources of useful info are welcome!

Briefly, a word on words used in this document:

"Bug" is used to generally denote the state of something being broken or incorrect. A "failure" is a bug in the observed behavior of one's code. An "infection" is a bug in the state of a particular object or variable. And finally, a "defect" is a bug in one's code. Generally, one or more defects is written into one's code, which, combined with particular program inputs, causes one or more infections that leads to a failure.

To find and fix the pesky bugs, follow the steps below.

1. Track the failure.
   1. Save one's work and the current state of the system. Check in the saved state via git with a descriptive message using the following format "[Bug] Insert 73-character-or-less bug description here."
   2. Create a github issue that documents the bug's existence. This will be a bug report containing:
      - A one line description of the problem. Should basically be the same as the message from the previous step.
      - A written description of the failure. What happened, in words?
      - A written description of the expected behavior. What was supposed to happen, in words?
      - Severity categorization: {Blocker; Major - aka large problem; Minor - aka an easy to fix problem; Enhancement - aka not a requirement but nice to have.}
      - A minimal working example or set of steps that reproduces the failure.
      - If possible (but not required now), a **simplified** test example that produces the same qualitative failure as the failure behavior you observed.
      - Any diagnostic information (errors, execution logs, etc.) reported when the failure was observed.
      - Environment information when the failure was observed (installed packages, operating system, memory capacity, etc.)
   3. Create a new branch off the feature branch to track work on fixing the bug.
   4. Open a pull request for the new branch to the original feature branch. Register that the issue is being worked on in this pull request by referencing the issue for the bug in the text body of the pull request on github.
   5. Add a test to the suite of regression tests to reproduce this **specific** software failure instance. Check the test in with git. This test differs from the simplified test above because it reproduces the exact bug that caught your attention in the first place.
      - See chapter 4, "Reproducing Problems" of "Why Programs Fail: A Guide to Systematic Debugging" by Morgan Kaufmann.
   6. Create an automated and simplified test case for this specific failure instance. While this step was optional before, it should definitely be performed now to ease one's later investigative efforts.
      - See chapter 5, "Simplifying Problems" of "Why Programs Fail: A Guide to Systematic Debugging" by Morgan Kaufmann.

2. Understand how the system that is producing the failure is supposed to work.
   1. Identify and read the basic documentation related to the (relevant?) external software tools being used at or around the point of failure.
   2. Take basic trainings (e.g. online tutorials, online courses, etc.) in the external tools being used in your project.
   3. Identify, store, and read the details about major / complex algorithms that one's software is implementing.

3. Understand how the system that is producing the failure is working.
   1. What is being produced during failure?

      Ensure that one has a debugging system in-place to identify and record the intermediate values being produced by one's software as well as which of those values are incorrect in the simplest created test case. In python, this is trivially provided by the built-in python debugger, but one should verify this condition for any non-python tools being used in the software.

   2. When does failure occur?

      If failure is intermittent or conditional, document what inputs, if any, cause the failure to occur rather than not.

   3. Where is the defective code?
      1. Hypothesize and record possible points where the bug could have been introduced.
         1. For each hypothesis, note its predicted observations. These observations will be tested for when attempting to falsify the hypothesis.
         2. For each hypothesis, note the empirical facts that suggest it.
         3. Set a time limit for this data and test-less work.
         4. Be sure to include logically likely contributors to the failure's existence (e.g. other known bugs, causes in program state / code / input, anomolies in the output or intermediate variables, poorly written code, etc.). The difference between these contributors and simply documenting the inputs that cause failur to occur is that the listed contributor should be a hypothesis describing what about the inputs makes one suspect the defect to be in a particular area of code.
      2. For each of those hypothesized points, determine whether the point is defective.
      3. For each defective point, determine whether that point actually changes the failure behavior.
      4. Once one's a-priori hypotheses have been exhausted (or once one's time limit for ad-hoc debugging has been reached), employ various brute-force / common search techniques to discover remaining causes of the failure. E.g.: [WIP--to be detailed.]
         1. Bisection Search
            1. Start at the beginning of a chunk of code where there is no code failure.
            2. Mark the earliest point in the code chunk where the failure has occurred.
            3. Go halfway between the two points and identify whether there are any infections (incorrect states) in the first half of the code chunk.
            4. If infections are found, repeat steps A to C on the code chunk between the original starting point and the earliest new infection that was found.
            5. If infections are not found, identify whether there are any infections in the second half of the code chunk.
            6. If infections are found in the second half of the code chunk, repeat steps A to C between the beginning point of the second half and the earliest new infection that was found.
            7. Cease when no more infections are located.
            8. Take all infections without causally preceding infections as defect sites.
         2. (Static / Dynamic) Ancestor Tracking
            1. Start at the infected value as reported by the code failure.
            2. Determine whether static or dynamic tracking will be used  (or both). Static tracking is based purely on the source code. Dynamic tracking is based on a particular execution with a given set of arguments.
            3. Identify the ancestors of the infection.
               1. Control ancestors: These objects or lines of code control or affect the execution of the code that created the current infection.
               2. Data ancestors: These objects or lines of code contain / create / affect the data used to create the current infection.
            4. Identify whether each of the ancestors are infected.
            5. Repeat steps A to D on each infected ancestor.
            6. Cease when there are no more ancestors.
            7. Collect all infected objects / variables without infected ancestors of their own as defect sites.
      5. Open a new task on the failure's github issue for each discovered contributor to the failure's existence (i.e.  each defect).
      6. Check out a new branch off of the bug-fixing branch for each discovered contributor to the failure's existence (i.e.  each defect).
      7. On the defect-fixing branch, create and check-in a test to document the defect's existence.
      8. Push each new branch to github and open a corresponding new pull-request (PR) for each discovered contributor to the failure's existence (i.e.  each defect). In the defect-fixing PRs, reference the failure's github issue, the test for the defect, and give a concise description of the defect.

    4. Why is the code defective?
       1. For each code defect identified above, hypothesize why the code is defective and record these hypotheses on the defect-fixing PR.
       2. Perform experiments to test one's hypotheses about the cause of the defect.
       3. Record the confirmed causes of the defect in the PR for the defect.
3. Fix the  failure.

   From the beginning of the chain of causation/computation to the end, isolate and fix each instance of defective code.

   1. For each defect being corrected, predict how a given change will (i) fix the specific failure case and (ii) fix the underlying cause of the defective code. Record these prediction in the defect's PR.
   2. Make a given change and test whether or not the defect is corrected.
   3. For each corrected defect, check whether the original failure is corrected by running the corresponding regression test.
   4. For each corrected defect, check whether any new problems are introduced by running the full test suite to ensure no new tests fail.
   5. For each corrected defect, check whether any similar defects exist in other parts of the code due to the same flawed reasoning that caused this defect.
   6. Close the PR for each corrected defect.
   7. Once the failure is fixed, check in the fix to git with a descriptive message following the format "[Bug-Fix] Insert 69-character-or-less description here."
   8. Merge the pull request to track that the bug was fixed and incorporate the changes.
