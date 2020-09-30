[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Build Status](https://github.com/AI-MEN/MLDiag/workflows/mldiag/badge.svg)
[![Coverage Status](https://codecov.io/gh/AI-MEN/MLDiag/branch/master/graph/badge.svg)](https://codecov.io/gh/AI-MEN/MLDiag) 
[![CodeFactor](https://www.codefactor.io/repository/github/AI-MEN/MLDiag/badge)](https://www.codefactor.io/repository/github/AI-MEN/MLDiag)

# MLDiag

This python library helps you diagnose machine learning models before deployment. 

Visit this introduction to understand about [MLDiag](https://github.com/AI-MEN/MLDiag/blob/master/blog/MLdiag.md). 


## Features
*   Generate synthetic data with adversarial attacks to evaluate model robustness
*   Make some interesting statistics on model behaviour
*   Simple, easy-to-use and lightweight library. Diagnose data in 3 lines of code
*   Plug and play to any neural network frameworks (e.g. PyTorch, TensorFlow) or standard machine learning framework (e.g. scikit-learn)
*   Support textual, image, audio and structured data
*   Can be added in a CI workflow
*   Can be used in command line or python scripts

    
## Quick Demo
*   [Example of diagnosis of Text classification model](https://github.com/AI-MEN/mldiag/blob/master/examples/tf_text_classification_diag.py)

## Quick start
The library supports python 3.7+ in linux and window platform.

To install the library:
```bash
pip install mldiag
```
or install the latest version (include BETA features) from github directly
```bash
pip install git+https://github.com/AI-MEN/mldiag.git
```

## Diagnostics
| Diagnostic | Target | Action | Description |
|:---:|:---:|:---:|:---:|
|Textual| Character | OCRError | Simulate ocr error |




## Recent Changes

See [changelog](https://github.com/AI-MEN/mldiag/blob/master/CHANGE.md) for more details.

## Extension Reading
*   [Data Augmentation library for Text](https://towardsdatascience.com/data-augmentation-library-for-text-9661736b13ff)
*   [Does your NLP model able to prevent adversarial attack?](https://medium.com/hackernoon/does-your-nlp-model-able-to-prevent-adversarial-attack-45b5ab75129c)
*   [How does Data Noising Help to Improve your NLP Model?](https://medium.com/towards-artificial-intelligence/how-does-data-noising-help-to-improve-your-nlp-model-480619f9fb10)
*   [Data Augmentation library for Speech Recognition](https://towardsdatascience.com/data-augmentation-for-speech-recognition-e7c607482e78)
*   [Data Augmentation library for Audio](https://towardsdatascience.com/data-augmentation-for-audio-76912b01fdf6)
*   [Unsupervied Data Augmentation](https://medium.com/towards-artificial-intelligence/unsupervised-data-augmentation-6760456db143)
*   [A Visual Survey of Data Augmentation in NLP](https://amitness.com/2020/05/data-augmentation-for-nlp/)


## Reference
This library uses:
* data (e.g. capturing from internet),
* research (e.g. following augmenter idea), 
* model (e.g. using pre-trained model) 

`TODO: update sources`
See [data source](https://github.com/AI-MEN/MLDiag/SOURCE.md) for more details.

## Citing

```latex
@misc{shabou2020mldiag,
  title={Machine learning diagnosis},
  author={Aymen SHABOU},
  howpublished={https://github.com/AI-MEN/MLDiag},
  year={2020}
}
```

## Contributions