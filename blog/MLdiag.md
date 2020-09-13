# MLDiag

an effective machine learing model diagnosing tool

## Introduction

Testing machine learning models is usually limited to some actions, mainly:
-   algorithmic performances (metrics) of the model over a validation  set (accuracy, precision, recall, etc.),
-   computational performances of the model (speed, memory, etc.)
-   examples where the model was most confidently incorrect,

Decision made to promote a model into production is usually based on theses  observations:
- is the current model offering an improvement over the existing version when evaluated on the same dataset.
- is the current model reaching the business required performance.


The model that fulfill this checklist is saved, along with hyperparamters and dataset that were used to validate the model, using MLFlow for instance. However, this traditional model checking lacks many important features:
- **where the model  does usually fail, and how to fix it?**
- **does the model generalize well on some variations over the evaluation set?**
- **how to track (and prevent) behavioral regressions for specific failure modes that had been previously addressed?** Indeed, you might improve the overall evaluation metric but introduce a regression on a critical subset of data. Or you could unknowingly add a gender bias to the model through the inclusion of a new dataset during training. 
- **how can we define the best directions for model future improvement** researcher usually look at a small set of incorrect predicted data to understand model failure. Such approach is biased since small samples  are likely unrepresentative of the true error distribution. A more precise, reproducible, scalable, and testable procedure is required.

The checklist could be larger, but work is usually made manually.

MLDiag is a simple, yet effective framework, to handle model evaluation automatically. It provides at the end of the evaluation a comprehensive report on model robustness to adversarial attacks and simple tracks to improve its performances.

## Related work

Recent works have tackled the model evaluation step.

Odena et al. introduced in this [paper](http://proceedings.mlr.press/v97/odena19a/odena19a.pdf) testing techniques for neural networks that can discover errors occurring only for rare inputs., called *Coverage-Guided Fuzzing (CGF)* methods, which are random mutations of inputs guided by a coverage metric toward the goal of satisfying user-specified constraints.
`TODO: read paper` 

Ribeiro et al. introduced in this [paper](https://homes.cs.washington.edu/~marcotcr/acl20_checklist.pdf) a task-agnostic methodology for testing NLP models, called *CheckList*.
`TODO: read paper` 

Jordan proposed in his [blog](https://www.jeremyjordan.me/testing-ml/?utm_campaign=Data_Elixir&utm_source=Data_Elixir_300) two behavior testing recipes to address the model testing step:
- *Pre-train tests* that allow us to identify some bugs early on and short-circuit a training job.
- *Post-train tests* use the trained model artifact to inspect behaviors for a variety of important scenarios that we define.

## MLDiag

MLDiag introduces several model tests to ensure its robustness and disposal to go to production.
We categorize theses tests into several categories and different scenario related to the model task.

### Descriptive
Descriptive tests highlight some model failures or success regarding dataset statistics.

| Test| Target(s)|  Description |  
|:---:|:---:|:---:|
|Length| Text| how data length impacts model errors (e.g. a text classification model always fails on short texts |
|Similar| text/Image| does the model make always the same error (e.g. a classification model that fails in a whole category or in a cluster of very similar data |


### Invariance
Inputs data are perturbed while measuring there impact on model prediction. This is closely related to data augmentation techniques, where we apply perturbations to inputs during training while preserving the original label. The problem with a `brut force` data augmentation is its computation cost. Using MLDiag tests, we can observe which perturbations affect the current model , we define then the set of augmentations that improve model consistency.

The list of model invariance tests is the following:
- Text tasks:
	- General classification sub-task:
		- Spelling mistake
		- Pronoun invariance
		- Sentence order invariance
	- Sentiment classification sub-task
		- 





## Further reading

**Blog posts**
Jeremy Jordan, [Effective testing for machine learning systems](https://www.jeremyjordan.me/testing-ml/?utm_campaign=Data_Elixir&utm_source=Data_Elixir_300), Aug 2020

**Papers**

-   [Beyond Accuracy: Behavioral Testing of NLP Models with CheckList](https://homes.cs.washington.edu/~marcotcr/acl20_checklist.pdf)
-   [TensorFuzz: Debugging Neural Networks with Coverage-Guided Fuzzing](http://proceedings.mlr.press/v97/odena19a/odena19a.pdf)
**Softwares**
- [Errudites](https://github.com/uwdata/errudite) an interactive tool for scalable, reproducible, and counterfactual error analysis of NLP models.

