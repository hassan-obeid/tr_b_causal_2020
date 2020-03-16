# Description
The dataset used in this project is taken from Brathwaite and Walker (2018).
```
Brathwaite, Timothy, and Joan L. Walker.
"Asymmetric, closed-form, finite-parameter models of multinomial choice."
Journal of choice modelling 29 (2018): 78-112.
```
See Appendix D (Section 11) of that paper for a general description of the dataset.

# Notes
Key facts and unexpected truths about this dataset to keep in mind are that:
1. All 'level-of-service' characteristics are based on the San Francisco [MTC travel skims](https://github.com/BayAreaMetro/modeling-website/wiki/SimpleSkims).
  This can have unexpected consequences.
  For instance, the distance skims do not define a travel distance between OD pairs for transit.
  (After all, one's distance may depend on what specific transit mode and route one takes).
  These missing values have been imputed in the dataset as zeros.
