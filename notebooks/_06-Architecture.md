---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.3.3
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Architecture


## 1. At a “high” level, how will the project deliver each of the needed requirements?
- **Try to think in terms of some manageable (e.g. 10-20) steps**
- **List needed intermediate products (e.g. pseudo-code, unit tests, etc.)**
- **Try to be concise but comprehensive?**






## 2. Description of major components of the system and their relations:





### 2.1 What should each component do?





### 2.2 What is the interface between each component and each other component?





### 2.3 How do components use each other (if they do?), and which ones are allowed to use which other components?





## 3. Description of needed development / computational environment:


### 3.1  Where will development take place (virtual environments? One’s laptop, etc.)?


- There are two alternatives:
  - **Virtual Machine**: Install JupyterHub on cloud server with multiple users and the same environment for each user. Collaborate directly on the Jupyter Notebooks and convert them to Jupytext .py files for easier code review and version control.
  - **Personal laptop**: work locally on personal laptop and push changes as progress is made to the right folder in the github repository.


### 3.2  How will one reliably construct the needed computational environment?


- There are two Alternatives:
  - If using **Virtual Machine**: The created JupyterHub server will have all the dependencies installed and available to all users on the server. The server administrator will be responsible for installing necessary environment from a .yml file.
  
  - If using **personal laptop**: 
      * Create a conda virtual environment with all the necessary packages for the project, and push it to the github repository as a .yml file in order for it to be duplicated by other members and end users, **OR**
      * Create a Docker image of the computational environment to be used by all team members.


### 3.3  How will one test that the computational environment is correctly created?


- Shell commands in either situation will be used to check that the correct environment was created/installed.
- Write class/function to make sure each of the needed packages and right python version is installed.


## 4. Description of major classes to be used (relates to major components above):
- **Class responsibilities**
- **Interactions between classes**
- **Class hierarchies / state transitions / object persistence**
- **Organization of classes into subsystems (if necessary / planned)**
- **Why does each class have “jurisdiction” over the parts of the system that it does?**





## 5. Description of the minimally viable product (if any):
- **Note the portions of the architecture responsible for producing the minimally viable product.**





## 6. Description of algorithms:


### 6.1  What, if any, computational algorithms are being implemented as part of this project instead of relying upon external implementations?





### 6.2 What purpose(s) do those algorithms serve?





### 6.3 Are each of these algorithms fully understood by someone on the project team?





### 6.4  What are alternative algorithms that could have served the purposes of the algorithms we are implementing?





### 6.5 Why are we using the particular algorithms we’ve chosen instead of others? 





### 6.6 Why do we believe the algorithms can be implemented?





### 6.7 Do we have any concerns with regard to algorithmic efficiency?





## 7. Description of data usage in the system:


### 7.1 How will data be ingested / accessed?





### 7.2 How will data be validated?





### 7.3 How will data be organized?





### 7.4 How will data be transformed?





### 7.5 How will created data be output and stored?





### 7.6 How will created data be validated?





### 7.7 How does all of the above differ online vs offline? (e.g. we may throw a critical error offline to prevent training with bad data but only log a warning online to note that the data in a request was unexpected.)





### 7.8 How will the flow / manipulation of data be controlled (and recorded / version-controlled) and made reproducible? E.g. data ingestion and manipulation to create training / testing sets.
- **Flyte?**
- **Make?**
- **Other?**  





## 8. Description of experimentation protocols (if relevant):
**If experiments need to be run (e.g. hyper-parameter tuning, model-selection, A/B tests, etc.):**


### 8.1 How will experiments be launched?





### 8.2 How will experiment reproducibility be ensured?





### 8.3 How will experiment meta-data (e.g. launch configurations) be stored?





### 8.4 How will experiment-created data (e.g. results) be stored?





### 8.5 How will experiment-created data be analyzed?





### 8.6 How will experiment analyses be prepared for public reporting? 





## 9. Description of user-interface:


### 9.1 How will users (including myself) interact with the system that is built?
- **Will there be a command line interface?**
- **Will users be editing configuration files, and if so, where will they be stored?**
- **Will there be ability to replay the DAG representing one’s data analysis?**





### 9.2 How will ease of changes be ensured (e.g. changing a given hyperparameter value in the source code of a model)?





### 9.3 Is the user-interface self-contained so that other parts of the system are insulated from changes in the user-interface?





## 10. Description of resource-management:


### 10.1 How will the system cope with large amounts of data during model training?





### 10.2 How much memory and time will it take the system to execute?
- **Offline / online prediction**
- **Model training**





### 10.3 How will the system interact with the external world to acquire resources? (e.g. get data or spawn virtual machines





#### 10.3.1 How does the system decide how much of an external resource is needed? (e.g. how many virtual CPUs are required during training?) 





## 11. Description of how the system will scale:


### 11.1 With increasing dataset sizes: 





### 11.2 With increasing numbers of models being used:





### 11.3 With increasing numbers of models being used:





### 11.4 With increasing numbers of parameters:





### 11.5 With increasing numbers of divergences/losses:





## 12. Description of how the system will interface with external systems:
- **E.g. if a model needs to be served in Go but it was trained in Python…**
- **If the system needs to manipulate Kubernetes for training...**





## 13. Description of how errors will be handled:


### 13.1 What are common expected errors from users and how can we guard against them?





### 13.2 Will we try to fix errors or merely notify users of the error’s presence?





### 13.3 When errors are encountered, will we quit immediately or wait until some specified point before notifying users of errors? For which errors is each strategy appropriate?





### 13.4 What are the conventions for error messages that the system reports?





### 13.5 Where are errors processed? At the point of detection, by a central error handling class, by functions above in the call stack, etc.





### 13.6 What level of responsibility does each class have in validating its own input? Is there a central class (or set of classes) that performs all validation? When can classes assume clean information?





### 13.7 How will we use (or not use) type hinting to help prevent errors?





## 14. Description of testing plan:


### 14.1 How will we test that all of the parts of the architecture are working correctly?
- **How will we create tests for each part of the architecture that are:**
 - **clear and unambiguous**
 - **capable of dealing with stochastic functions / objects**
 - **minimizing reliance / use of stochasticity**
 - **fast to evaluate**


### 14.2 What are the various types of tests that are needed?





### 14.3 Are there any parts of the architecture that we do not know how to test? How will these knowledge gaps be resolved before the coding begins?





### 14.4 How will discovered bugs be turned into test cases?





### 14.5 How will the software be reviewed in general (beyond automated tests) to reduce the probability of programming bugs into the code / increase the chance of producing correctly functioning software?





## 15. Description of project-sharing procedures:


### 15.1 Is there a plan in place for producing documentation for the software?
- **Documentation for operating the software**
- **Documentation of software limits**
- **Usage examples and tutorials**
- **Key project learnings / retrospective notes**

Documentation, in the form of Jupyter Notebook accompanied by a Markdown file (README.md) detailing the procedure for installing the environment and the content of folders and notebooks. Any limitations to the project will be noted in the jupyter notebooks developed to illustrate the analysis effort. The README.md markdown file will also will point to documentation pages for any packages used in the project.


### 15.2 Is there a plan for sharing the software with others?
- **File upload?**
- **Upload to package repository?**

All project related material including code, documentation, presentation, environment files, and datasets will be uploaded to the project github repository.


### 15.3 Will the project have an associated website? If so is there a plan for its development?

The only "website" for the project will be its github repository.


### 15.4 Is there a plan for producing any written or visual materials that others may wish to view?
- **Final or interim project reports**
- **Tables of key project results**
- **Graphs, diagrams, images from the project, e.g.:**
 - **Plots of system performance**
 - **Model checking plots**


## 16 Description of maintenance plan:


### 16.1 How will the system / software be maintained after initial creation?





### 16.2 What needs to be done for maintenance?





### 16.3 Who will perform needed maintenance activities?





## 17. Description of project tools:


### 17.1 What are all the tools and major external systems upon which the project depends?





### 17.2 Is there evidence that we understand the basics of how all of those tools work?
- **Have we read any/all relevant documentation and instruction materials?**
- **Have we taken any basic training courses in any of these tools?**





## 18. Description of project risks and mitigation efforts:


### 18.1  There should be some evidence for, and of, project feasibility:
- **How do we know that every part of the architecture will work?**
- **Based on the architecture, what are all needed resources?**





### 18.2 What could render the project infeasible?
- **Is there adequate resourcing?**





### 18.3 Are there known sources of difficulty based on the planned tasks?





### 18.4 Are there mismatches between prior knowledge and required tasks?





### 18.5 What are the weakest areas of the architecture?
- **Where is the project most vulnerable to failure?**
- **How are these weaknesses being addressed?**





## 19. Description of the approach to over-engineering:


### To what extent should planning be done to prevent all errors versus to produce the simplest system necessitated by the requirements?





## 20.  Description of change strategy:


### 20.1 What steps are being taken to ensure that the architecture is able to handle changes with maximal ease / flexibility? e.g.:
- **Newly desired capabilities based on usage of initial iterations**
- **Adding planned features that didn’t make it to the minimal viable product**
- **Allowing for easy change of third party dependencies**
- **Complying with any relevant standards for maximal compatibility with outside products**
- **New ways of interaction with users (e.g. avoiding type checking as much as possible in favor of duck-typing)**
- **Are classes maximally independent such that changes in one class don’t necessitate changes in the others?**
- **Do we have a plan in place for test changes? For example if needing to change the data used in a test, can that be easily done without having to rewrite the test? E.g. data stored outside the code, or integration tests that don’t test for a specific prediction output but rather a property of the prediction outputs so that the model can change without the test necessitating a rewrite.**






### 20.2 What is the strategy for handling requests for new features from or updates of the system after its built?





### 20.3 How will bug-fixes be handled? By whom?





## 21.  Description of reuse strategy:


### How is the architecture maximally leveraging previously completed work?





## 22. Description of system alternatives (i.e. alternatives to the project as a whole):


### 22.1 What are the closest alternative systems that could be used for this project or for parts of the project?
- **How can those alternate systems be leveraged?**


### 22.2 Why is it expected that this project’s custom built sub-systems will be better than the available alternatives for those sub-systems?





## 23. Description of architectural alternatives (i.e. alternatives to how to carry out the project):


### 23.1 What are the closest alternative systems that could be used for this project or for parts of the project?
- **How can those alternate systems be leveraged?**





### 23.2 Why is it expected that this project’s custom built sub-systems will be better than the available alternatives for those sub-systems?





## 24. Description of architectural alternatives (i.e. alternatives to how to carry out the project):


### 24.1 What were alternative ways to architect the system to satisfy the requirements?
- **If there were no other alternatives, then why were there no other alternatives and can we prove this was really the case?**





### 24.2 Why did we choose to architect the system the way we did?





#### 24.2.1 What were our objectives in choosing the given architecture?





#### 24.2.2 What were the motivations for all major decisions?


## 25. Description of expected debugging protocol:
### Should detail the expected debugging process to be used when, despite all of our planning and checking, failure is experienced while using some piece of the implemented architecture.





## 26. Final checks:
### Do any parts of the system seem over- or under-architected. Empirically, are the descriptions of any sub-component much more thorough than any other?



