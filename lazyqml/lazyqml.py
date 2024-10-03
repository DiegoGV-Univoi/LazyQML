


class QuantumClassifier():
    """
    This module helps in fitting to all the classification algorithms that are available in Scikit-learn
    Parameters
    ----------
    verbose : bool, optional (default=False)
        Verbose True for showing every training message during the fit.
    ignoreWarnings : bool, optional (default=True)
        When set to True, the warning related to algorigms that are not able to run are ignored.
    customMetric : function, optional (default=None)
        When function is provided, models are evaluated based on the custom evaluation metric provided.
    customImputerNum : function, optional (default=None)
        When function is provided, models are imputed based on the custom numeric imputer provided.
    customImputerCat : function, optional (default=None)
        When function is provided, models are imputed based on the custom categorical imputer provided.
    prediction : bool, optional (default=False)
        When set to True, the predictions of all the models models are returned as a pandas dataframe.
    classifiers : list of strings, optional (default=["all"])
        When function is provided, trains the chosen classifier(s) ["all", "qsvm", "qnn", "qnnbag"].
    embeddings : list of strings, optional (default=["all"])
        When function is provided, trains the chosen embeddings(s) ["all", "amplitude_embedding", "ZZ_embedding", "rx_embedding", "rz_embedding", "ry_embedding"].
    ansatzs : list of strings, optional (default=["all"])
        When function is provided, trains the chosen ansatzs(s) ["all", "HPzRx", "tree_tensor", "two_local", "hardware_efficient"].
    randomSate : int, optional (default=1234)
        This integer is used as a seed for the repeatability of the experiments.
    nqubits : int, optional (default=8)
        This integer is used for defining the number of qubits of the quantum circuits that the models will use.
    numLayers : int, optional (default=5)
        The number of layers that the Quantum Neural Network (QNN) models will use, is set to 5 by default.
    numPredictors : int, optional (default=10)
        The number of different predictoras that the Quantum Neural Networks with Bagging (QNN_Bag) will use, is set to 10 by default.
    learningRate : int, optional (default=0.01)
        The parameter that will be used for the optimization process of all the Quantum Neural Networks (QNN) in the gradient descent, is set to 0.01 by default.
    epochs : int, optional (default=100)
        The number of complete passes that will be done over the dataset during the fitting of the models.
    runs : int, optional (default=1)
        The number of training runs that will be done with the Quantum Neural Network (QNN) models.
    maxSamples : float, optiona (default=1.0)
        A floating point number between 0 and 1.0 that indicates the percentage of the dataset that will be used for each Quantum Neural Network with Bagging (QNN_Bag).

    Examples
    --------
    >>> from lazyqml.supervised import QuantumClassifier
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.model_selection import train_test_split
    >>> data = load_breast_cancer()
    >>> X = data.data
    >>> y= data.target
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.5,random_state =123)
    >>> clf = QuantumClassifier(verbose=0,ignore_warnings=True, customMetric=None)
    >>> models,predictions = clf.fit(X_train, X_test, y_train, y_test)
    >>> model_dictionary = clf.provide_models(X_train,X_test,y_train,y_test)
    >>> models
    | Model       | Embedding           | Ansatz             |   Accuracy |   Balanced Accuracy |   ROC AUC |   F1 Score |   Time taken |
    |:------------|:--------------------|:-------------------|-----------:|--------------------:|----------:|-----------:|-------------:|
    | qsvm        | amplitude_embedding | ~                  |   0.807018 |            0.782339 |  0.782339 |   0.802547 |     43.7487  |
    | qnn         | amplitude_embedding | hardware_efficient |   0.77193  |            0.743218 |  0.743218 |   0.765533 |      7.92101 |
    | qnn         | ry_embedding        | hardware_efficient |   0.71345  |            0.689677 |  0.689677 |   0.709295 |      8.00107 |
    | qnn         | rz_embedding        | HPzRx              |   0.707602 |            0.687322 |  0.687322 |   0.704992 |      1.61301 |
    | qnn         | rx_embedding        | hardware_efficient |   0.707602 |            0.687322 |  0.687322 |   0.704992 |      7.95751 |
    | qnn         | rz_embedding        | hardware_efficient |   0.707602 |            0.677327 |  0.677327 |   0.700129 |      7.7675  |
    | qnn         | amplitude_embedding | HPzRx              |   0.707602 |            0.677327 |  0.677327 |   0.700129 |      1.84443 |
    | qnn         | ry_embedding        | HPzRx              |   0.690058 |            0.672758 |  0.672758 |   0.688787 |      1.78982 |
    | qnn         | rx_embedding        | HPzRx              |   0.701754 |            0.662479 |  0.662479 |   0.688319 |      1.74136 |
    | qnn         | rx_embedding        | two_local          |   0.672515 |            0.648201 |  0.648201 |   0.668402 |      1.97186 |
    | qsvm        | ry_embedding        | ~                  |   0.684211 |            0.647915 |  0.647915 |   0.672813 |     42.0951  |
    | qsvm        | rx_embedding        | ~                  |   0.684211 |            0.647915 |  0.647915 |   0.672813 |     42.0835  |
    | qsvm        | rz_embedding        | ~                  |   0.684211 |            0.647915 |  0.647915 |   0.672813 |     42.0728  |
    | qnn         | amplitude_embedding | two_local          |   0.660819 |            0.645988 |  0.645988 |   0.660819 |      1.862   |
    | qnn_bag_0.5 | amplitude_embedding | tree_tensor        |   0.684211 |            0.642918 |  0.642918 |   0.668975 |      2.29442 |
    | qnn         | rx_embedding        | tree_tensor        |   0.654971 |            0.641134 |  0.641134 |   0.655388 |      1.44079 |
    | qnn         | rz_embedding        | two_local          |   0.660819 |            0.630997 |  0.630997 |   0.653742 |      1.92491 |
    | qnn         | rz_embedding        | tree_tensor        |   0.637427 |            0.629069 |  0.629069 |   0.639648 |      1.43276 |
    | qnn         | ry_embedding        | tree_tensor        |   0.649123 |            0.628784 |  0.628784 |   0.647148 |      1.49096 |
    | qnn         | amplitude_embedding | tree_tensor        |   0.643275 |            0.623929 |  0.623929 |   0.641812 |      1.48968 |
    | qnn_bag_0.8 | ry_embedding        | HPzRx              |   0.666667 |            0.613364 |  0.613364 |   0.639261 |      3.05536 |
    | qnn_bag_0.8 | ry_embedding        | two_local          |   0.643275 |            0.601442 |  0.601442 |   0.627205 |      3.4717  |
    | qnn_bag_0.8 | amplitude_embedding | tree_tensor        |   0.614035 |            0.592162 |  0.592162 |   0.611863 |      2.0676  |
    | qnn_bag_0.8 | amplitude_embedding | two_local          |   0.614035 |            0.589663 |  0.589663 |   0.61059  |      2.88992 |
    | qnn_bag_0.8 | amplitude_embedding | HPzRx              |   0.625731 |            0.58438  |  0.58438  |   0.610027 |      3.10873 |
    | qnn_bag_0.8 | amplitude_embedding | hardware_efficient |   0.625731 |            0.579383 |  0.579383 |   0.605158 |     10.7474  |
    | qnn_bag_0.5 | amplitude_embedding | hardware_efficient |   0.619883 |            0.577027 |  0.577027 |   0.602759 |     10.5239  |
    | qnn_bag_0.8 | rx_embedding        | two_local          |   0.608187 |            0.574814 |  0.574814 |   0.599111 |      3.0376  |
    | qnn_bag_0.5 | amplitude_embedding | HPzRx              |   0.614035 |            0.574672 |  0.574672 |   0.600105 |      2.94314 |
    | qnn         | ry_embedding        | two_local          |   0.590643 |            0.572744 |  0.572744 |   0.590643 |      2.03271 |
    | qnn_bag_0.3 | ry_embedding        | two_local          |   0.614035 |            0.564677 |  0.564677 |   0.590048 |      2.89546 |
    | qnn_bag_0.5 | ry_embedding        | two_local          |   0.614035 |            0.564677 |  0.564677 |   0.590048 |      2.87825 |
    | qnn_bag_0.3 | rx_embedding        | hardware_efficient |   0.614035 |            0.564677 |  0.564677 |   0.590048 |     10.398   |
    | qnn_bag_0.5 | rx_embedding        | hardware_efficient |   0.614035 |            0.564677 |  0.564677 |   0.590048 |     10.4352  |
    | qnn_bag_0.5 | rz_embedding        | hardware_efficient |   0.608187 |            0.559823 |  0.559823 |   0.585266 |     10.5017  |
    | qnn_bag_0.3 | rz_embedding        | hardware_efficient |   0.608187 |            0.559823 |  0.559823 |   0.585266 |     10.459   |
    | qnn_bag_0.8 | rz_embedding        | two_local          |   0.608187 |            0.559823 |  0.559823 |   0.585266 |      2.90858 |
    | qnn_bag_0.5 | ry_embedding        | HPzRx              |   0.602339 |            0.557467 |  0.557467 |   0.583154 |      3.34753 |
    | qnn_bag_0.3 | ry_embedding        | HPzRx              |   0.602339 |            0.557467 |  0.557467 |   0.583154 |      2.8865  |
    | qnn_bag_0.8 | rx_embedding        | hardware_efficient |   0.614035 |            0.557182 |  0.557182 |   0.580603 |     10.668   |
    | qnn_bag_0.8 | rz_embedding        | tree_tensor        |   0.578947 |            0.55554  |  0.55554  |   0.576578 |      2.0455  |
    | qnn_bag_0.5 | rz_embedding        | tree_tensor        |   0.578947 |            0.55554  |  0.55554  |   0.576578 |      2.01763 |
    | qnn_bag_0.3 | rz_embedding        | tree_tensor        |   0.578947 |            0.55554  |  0.55554  |   0.576578 |      2.42137 |
    | qnn_bag_0.3 | ry_embedding        | hardware_efficient |   0.602339 |            0.554969 |  0.554969 |   0.58048  |     10.53    |
    | qnn_bag_0.5 | ry_embedding        | hardware_efficient |   0.602339 |            0.554969 |  0.554969 |   0.58048  |     10.0354  |
    | qnn_bag_0.5 | rz_embedding        | HPzRx              |   0.608187 |            0.554826 |  0.554826 |   0.579267 |      2.89535 |
    | qnn_bag_0.3 | rz_embedding        | HPzRx              |   0.608187 |            0.554826 |  0.554826 |   0.579267 |      3.23855 |
    | qnn_bag_0.3 | ry_embedding        | tree_tensor        |   0.608187 |            0.554826 |  0.554826 |   0.579267 |      2.06946 |
    | qnn_bag_0.8 | rz_embedding        | hardware_efficient |   0.608187 |            0.554826 |  0.554826 |   0.579267 |     10.5748  |
    | qnn_bag_0.5 | ry_embedding        | tree_tensor        |   0.608187 |            0.554826 |  0.554826 |   0.579267 |      2.06709 |
    | qnn_bag_0.8 | ry_embedding        | tree_tensor        |   0.608187 |            0.554826 |  0.554826 |   0.579267 |      2.49849 |
    | qnn_bag_0.8 | ry_embedding        | hardware_efficient |   0.608187 |            0.552327 |  0.552327 |   0.575973 |     10.6611  |
    | qnn_bag_0.8 | rz_embedding        | HPzRx              |   0.584795 |            0.5504   |  0.5504   |   0.575177 |      2.91932 |
    | qnn_bag_0.3 | rx_embedding        | HPzRx              |   0.590643 |            0.547758 |  0.547758 |   0.573467 |      3.04813 |
    | qnn_bag_0.5 | rx_embedding        | HPzRx              |   0.590643 |            0.547758 |  0.547758 |   0.573467 |      2.8814  |
    | qnn_bag_0.8 | rx_embedding        | HPzRx              |   0.608187 |            0.54733  |  0.54733  |   0.56875  |      3.32461 |
    | qnn_bag_0.5 | amplitude_embedding | two_local          |   0.573099 |            0.535694 |  0.535694 |   0.561126 |      2.8526  |
    | qnn_bag_0.8 | rx_embedding        | tree_tensor        |   0.584795 |            0.530411 |  0.530411 |   0.554148 |      2.35702 |
    | qnn_bag_0.5 | rz_embedding        | two_local          |   0.584795 |            0.530411 |  0.530411 |   0.554148 |      2.83122 |
    | qnn_bag_0.3 | rz_embedding        | two_local          |   0.584795 |            0.530411 |  0.530411 |   0.554148 |      3.22892 |
    | qnn_bag_0.5 | rx_embedding        | two_local          |   0.584795 |            0.530411 |  0.530411 |   0.554148 |      3.22437 |
    | qnn_bag_0.3 | rx_embedding        | two_local          |   0.584795 |            0.530411 |  0.530411 |   0.554148 |      2.85127 |
    | qnn_bag_0.5 | rx_embedding        | tree_tensor        |   0.584795 |            0.530411 |  0.530411 |   0.554148 |      2.05024 |
    | qnn_bag_0.3 | rx_embedding        | tree_tensor        |   0.584795 |            0.530411 |  0.530411 |   0.554148 |      2.18715 |
    | qnn_bag_0.8 | ZZ_embedding        | hardware_efficient |   0.590643 |            0.515277 |  0.515277 |   0.523392 |     10.3795  |
    | qnn         | ZZ_embedding        | HPzRx              |   0.508772 |            0.504783 |  0.504783 |   0.513787 |      1.94063 |
    | qnn_bag_0.8 | ZZ_embedding        | tree_tensor        |   0.602339 |            0.5      |  0.5      |   0.452854 |      2.57792 |
    | qnn_bag_0.5 | ZZ_embedding        | hardware_efficient |   0.602339 |            0.5      |  0.5      |   0.452854 |     10.6524  |
    | qnn_bag_0.3 | ZZ_embedding        | hardware_efficient |   0.602339 |            0.5      |  0.5      |   0.452854 |     10.6306  |
    | qnn_bag_0.5 | ZZ_embedding        | two_local          |   0.602339 |            0.5      |  0.5      |   0.452854 |      2.86801 |
    | qnn_bag_0.3 | ZZ_embedding        | two_local          |   0.602339 |            0.5      |  0.5      |   0.452854 |      2.87458 |
    | qsvm        | ZZ_embedding        | ~                  |   0.602339 |            0.5      |  0.5      |   0.452854 |     43.0182  |
    | qnn_bag_0.3 | ZZ_embedding        | HPzRx              |   0.602339 |            0.5      |  0.5      |   0.452854 |      2.89995 |
    | qnn_bag_0.5 | ZZ_embedding        | HPzRx              |   0.602339 |            0.5      |  0.5      |   0.452854 |      3.35156 |
    | qnn_bag_0.3 | amplitude_embedding | tree_tensor        |   0.602339 |            0.5      |  0.5      |   0.452854 |      2.16821 |
    | qnn_bag_0.3 | ZZ_embedding        | tree_tensor        |   0.602339 |            0.5      |  0.5      |   0.452854 |      2.1247  |
    | qnn_bag_0.5 | ZZ_embedding        | tree_tensor        |   0.602339 |            0.5      |  0.5      |   0.452854 |      2.07843 |
    | qnn_bag_0.8 | ZZ_embedding        | two_local          |   0.590643 |            0.49279  |  0.49279  |   0.457223 |      3.03675 |
    | qnn_bag_0.8 | ZZ_embedding        | HPzRx              |   0.561404 |            0.478512 |  0.478512 |   0.473343 |      3.07628 |
    | qnn_bag_0.3 | amplitude_embedding | hardware_efficient |   0.532164 |            0.474229 |  0.474229 |   0.495696 |     10.6454  |
    | qnn         | ZZ_embedding        | tree_tensor        |   0.48538  |            0.470374 |  0.470374 |   0.488533 |      1.41871 |
    | qnn_bag_0.3 | amplitude_embedding | HPzRx              |   0.526316 |            0.466876 |  0.466876 |   0.48737  |      3.63382 |
    | qnn_bag_0.3 | amplitude_embedding | two_local          |   0.526316 |            0.466876 |  0.466876 |   0.48737  |      3.10482 |
    | qnn         | ZZ_embedding        | hardware_efficient |   0.467836 |            0.463307 |  0.463307 |   0.473372 |      8.22384 |
    | qnn         | ZZ_embedding        | two_local          |   0.461988 |            0.455954 |  0.455954 |   0.467481 |      2.13294 |
    """

    def __init__(self, nqubits=8, randomstate=1234, predictions=False, ignoreWarnings=True, numPredictors=10, numLayers=5, customMetric=None, customImputerNum=None, customImputerCat=None, classifiers=["all"],ansatzs=["all"],embeddings=["all"],features=[0.3,0.5,0.8],verbose=False,learningRate=0.01,epochs=100,runs=1,maxSamples=1.0):
        pass

    def fit(self, X_train, y_train, X_test, y_test,showTable=True):
        pass

    def repeated_cross_validation(self, X, y, n_splits=5, n_repeats=10, showTable=True):
        pass

    def leave_one_out(self, X, y, showTable=True):
        pass
