++ The notebook 'Assignment_3_15CS10053.ipynb' is written for Python-3.
++ It uses training and testing files in 'conllu' format. Their system path must be specified in the notebook. Please modify the path in the second notebook cell accordingly.
++ The libraries required are imported in the first cell. Please ensure they are installed on the system.
++ The cells can be run sequentially. Binary model files are created by training and tested to give LAS and UAS for different combination of Features and Classifiers.
++ In the notebook, scores are reported by removing each feature separately. Therefore, we report 5 scores -- one including all 4 features and 4 others by excluding each feature separately.
++ To check for other feature combinations, simply modify the 'include_features' dict and pass it to the modifiedTransitionParser.train() method.
++ The classifier can also be specified in the call to the train method. (svm, logistic_reg, mlp)
