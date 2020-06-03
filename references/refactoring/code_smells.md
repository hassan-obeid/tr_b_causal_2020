# Code smells and testing needs

## Needed tests
- Unit Tests  
What must be true, at a painstakingly obvious level, in order for one to believe that a method or function's basic functionality is correctly implemented, given specially chosen test cases?
- Integration tests  
Does the overall functionality of one’s entire project or subset of project work?
- Property-based tests  
Does one code satisfy required properties needed for correct implementation?
   - Smoke Tests  
   Test that the basic & critical functions always work. E.g.:
      - Given a valid input, a classifier should always predict a class.
      - Given a valid training set, the classifier should always successfully complete training.
    - Metamorphic Tests  
       - Look for invariances that MUST hold for one’s code as one morphs / changes the input. E.g.:
          - sin(12) = sin(pi - 12)
          -  the ratio of conditional distributions should equal the ratio of joint distributions for ANY values used in numerator and denominator of the conditional distributions.
       - Look for relationships that MUST hold for one’s code as one morphs / changes the input [E.g f(x) = a + c * x, then f(b * x) > f(x) for positive c].
- Regression Tests  
Make sure that bugs that were previously encounted never arise again. Test against minimal working examples from previous bug cases.

## Code smells
- Are all variable names meaningful and intention-revealing?
   - No single letter names
   - No names that are purely greek letters (e.g. alpha)
- Identify the type of design pattern being used, and favor object-oriented design over procedural design unless one has ample reason to do otherwise:
   - Procedural design will typically separate data from all processing procedures.
   - Object oriented design will mix data and processing procedures in a single object, but partition multiple objects.
   - Signs of the problem:
      - Single class with many attributes / methods / or both
      - Single controller with associated simple, data-object classes
      - A single class encapsulates most of the entire functionality for one’s project / application.
      - The class is hard to test.
- Identify code that is too long/ large.
   - Methods / Functions longer than 100 lines
   - Classes with  many (more than 8 or so?) attributes
   - Classes with many methods
   - Methods with long parameter lists (especially if many of these parameters appear in multiple methods)
   - A class that does nothing but store data
- Code that doesn’t use enough OO-programming
   - Excessive use of if-statements
   - Lots of temporary fields or fields that are only filled at certain times.
   - Subclasses that don’t use many of the parent class attributes or methods.
- Code that cannot be changed without having to change code in multiple places.
   - Changing functionality of a given method / function requires you to change many other methods on a given class.
   - Changing a class’ behavior requires you to change many methods on another class.
   - Having to create parallel subclasses in order to subclass a class of interest.
   - Having to change the behavior of multiple classes
- Unnecessary code
   - Code that cannot be understood without comments.
   - Duplicated code
   - A class that only stores data
   - Code paths that are never reached.
   - A class that is excessively tiny (only does one thing). Functions / methods should do one thing, not classes.
   - Unused classes / methods / fields / or parameters
   - Classes that perform the same function but with different method names.
- Inappropriate splitting of responsibility
   - A class having to access data from another object more than one’s own data.
   - A class using the internal methods of another class.
   - Excessively long chains of method calls between objects
   - Classes that do nothing but delegate to other classes.
