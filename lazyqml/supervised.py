#======================================================================================
# Main Module
#======================================================================================

"""
 Import modules
"""

from common import *
import warnings
warnings.filterwarnings("ignore")

"""
 Classifiers
"""

ansatzs = ['hardware_efficient','tree_tensor','two_local','HPzRx']

embeddings = ['rx_embedding','ry_embedding','rz_embedding','ZZ_embedding', 'amplitude_embedding']

features = [0.8, 0.5, 0.3]

"""
 Quantum Classifier
"""

class QuantumClassifier():
    """
    This module helps in fitting to all the classification algorithms that are available in Scikit-learn
    Parameters
    ----------
    verbose : int, optional (default=0)
        For the liblinear and lbfgs solvers set verbose to any positive number for verbosity.
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
    classifiers : string, optional (default="all")
        When function is provided, trains the chosen classifier(s) ["all", "qsvm", "qnn", "qnnbag"].
    randomSate : int, optional (default=1234)
        This integer is used as a seed for the repeatability of the experiments.
    nqubits : int, optional (default=8)
        This integer is used for defining the number of qubits of the quantum circuits that the models will use.
    numLayers : int, optional (default=5)
        The number of layers that the Quantum Neural Network (QNN) models will use, is set to 5 by default.
    numPredictors : int, optional (default=10)
        The number of different predictoras that the Quantum Neural Networks with Bagging (QNN_Bag) will use, is set to 10 by default.
    learningRate : int, optional (default=0.1)
        The parameter that will be used for the optimization process of all the Quantum Neural Networks (QNN) in the gradient descent, is set to 0.1 by default.
    optimizer : optax optimizer, optional (default=optax.adam(learningRate))
        The function that will be used during the gradient descent optimization of the trainable parameters, this must be an optax optimizer function.
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
    | Model       | Embedding           | Ansatz             |   Accuracy |   Balanced Accuracy | ROC AUC   |   F1 Score |   Time taken |
    |:------------|:--------------------|:-------------------|-----------:|--------------------:|:----------|-----------:|-------------:|
    | qsvm        | rx_embedding        |                    |   0.955556 |            0.960784 |           |  0.956187  |      3.00024 |
    | qsvm        | rz_embedding        |                    |   0.955556 |            0.960784 |           |  0.956187  |      2.93275 |
    | qsvm        | ry_embedding        |                    |   0.955556 |            0.960784 |           |  0.956187  |      2.98143 |
    | qnn         | amplitude_embedding | hardware_efficient |   0.733333 |            0.754248 |           |  0.723993  |     11.207   |
    | qnn         | ZZ_embedding        | two_local          |   0.733333 |            0.754248 |           |  0.723993  |      2.9692  |
    | qnn         | rz_embedding        | two_local          |   0.733333 |            0.754248 |           |  0.723993  |      2.94752 |
    | qnn         | rx_embedding        | two_local          |   0.733333 |            0.754248 |           |  0.723993  |      3.07974 |
    | qnn         | ry_embedding        | two_local          |   0.733333 |            0.754248 |           |  0.723993  |      2.94321 |
    | qnn         | amplitude_embedding | HPzRx              |   0.733333 |            0.753159 |           |  0.717882  |      2.32015 |
    | qsvm        | amplitude_embedding |                    |   0.711111 |            0.734641 |           |  0.697691  |      3.99333 |
    | qnn         | rx_embedding        | HPzRx              |   0.666667 |            0.681699 |           |  0.650911  |      2.2885  |
    | qnn         | ZZ_embedding        | HPzRx              |   0.666667 |            0.681699 |           |  0.650911  |      2.09387 |
    | qnn         | rz_embedding        | HPzRx              |   0.666667 |            0.681699 |           |  0.650911  |      2.31533 |
    | qnn         | ry_embedding        | HPzRx              |   0.666667 |            0.681699 |           |  0.650911  |      2.29481 |
    | qnn_bag_0.8 | amplitude_embedding | tree_tensor        |   0.622222 |            0.676471 |           |  0.609367  |      5.76442 |
    | qnn         | amplitude_embedding | two_local          |   0.622222 |            0.663834 |           |  0.610864  |      2.72723 |
    | qnn_bag_0.8 | amplitude_embedding | HPzRx              |   0.6      |            0.657952 |           |  0.586446  |      3.39962 |
    | qnn         | rz_embedding        | hardware_efficient |   0.622222 |            0.656209 |           |  0.580471  |     11.0559  |
    | qnn         | ZZ_embedding        | hardware_efficient |   0.622222 |            0.656209 |           |  0.580471  |     11.0533  |
    | qnn         | rx_embedding        | hardware_efficient |   0.622222 |            0.656209 |           |  0.580471  |     11.5151  |
    | qnn         | ry_embedding        | hardware_efficient |   0.622222 |            0.656209 |           |  0.580471  |     11.1438  |
    | qnn_bag_0.8 | amplitude_embedding | two_local          |   0.6      |            0.644227 |           |  0.589402  |      3.95424 |
    | qnn_bag_0.8 | amplitude_embedding | hardware_efficient |   0.6      |            0.643137 |           |  0.591346  |     13.716   |
    | qnn         | ZZ_embedding        | tree_tensor        |   0.6      |            0.636601 |           |  0.547036  |      4.11366 |
    | qnn         | rz_embedding        | tree_tensor        |   0.6      |            0.636601 |           |  0.547036  |      4.28233 |
    | qnn         | ry_embedding        | tree_tensor        |   0.6      |            0.636601 |           |  0.547036  |      4.25941 |
    | qnn         | rx_embedding        | tree_tensor        |   0.6      |            0.636601 |           |  0.547036  |      4.38062 |
    | qsvm        | ZZ_embedding        |                    |   0.622222 |            0.636383 |           |  0.641239  |      3.32188 |
    | qnn         | amplitude_embedding | tree_tensor        |   0.577778 |            0.616993 |           |  0.511375  |      4.40494 |
    | qnn_bag_0.5 | amplitude_embedding | HPzRx              |   0.488889 |            0.508279 |           |  0.48642   |      3.33889 |
    | qnn_bag_0.5 | amplitude_embedding | tree_tensor        |   0.444444 |            0.471242 |           |  0.410935  |      6.13786 |
    | qnn_bag_0.5 | amplitude_embedding | hardware_efficient |   0.444444 |            0.471242 |           |  0.410935  |     13.5337  |
    | qnn_bag_0.5 | amplitude_embedding | two_local          |   0.422222 |            0.452723 |           |  0.372047  |      3.86997 |
    | qnn_bag_0.8 | rz_embedding        | HPzRx              |   0.4      |            0.399129 |           |  0.423165  |      3.32705 |
    | qnn_bag_0.8 | ZZ_embedding        | HPzRx              |   0.4      |            0.399129 |           |  0.423165  |      3.50388 |
    | qnn_bag_0.8 | rx_embedding        | hardware_efficient |   0.4      |            0.399129 |           |  0.423165  |     14.2781  |
    | qnn_bag_0.8 | ry_embedding        | HPzRx              |   0.4      |            0.399129 |           |  0.423165  |      4.21789 |
    | qnn_bag_0.8 | ry_embedding        | hardware_efficient |   0.4      |            0.399129 |           |  0.423165  |     13.5693  |
    | qnn_bag_0.8 | rx_embedding        | HPzRx              |   0.4      |            0.399129 |           |  0.423165  |      3.30752 |
    | qnn_bag_0.8 | ZZ_embedding        | hardware_efficient |   0.4      |            0.399129 |           |  0.423165  |     13.6998  |
    | qnn_bag_0.8 | rz_embedding        | hardware_efficient |   0.4      |            0.399129 |           |  0.423165  |     13.7134  |
    | qnn_bag_0.5 | rz_embedding        | hardware_efficient |   0.333333 |            0.390196 |           |  0.275556  |     13.5766  |
    | qnn_bag_0.5 | ZZ_embedding        | HPzRx              |   0.333333 |            0.390196 |           |  0.275556  |      3.42679 |
    | qnn_bag_0.5 | ry_embedding        | two_local          |   0.333333 |            0.390196 |           |  0.275556  |      4.44122 |
    | qnn_bag_0.5 | rx_embedding        | tree_tensor        |   0.333333 |            0.390196 |           |  0.275556  |      5.3968  |
    | qnn_bag_0.5 | rx_embedding        | hardware_efficient |   0.333333 |            0.390196 |           |  0.275556  |     13.53    |
    | qnn_bag_0.5 | ZZ_embedding        | two_local          |   0.333333 |            0.390196 |           |  0.275556  |      4.51936 |
    | qnn_bag_0.5 | ZZ_embedding        | tree_tensor        |   0.333333 |            0.390196 |           |  0.275556  |      5.96975 |
    | qnn_bag_0.5 | rz_embedding        | HPzRx              |   0.333333 |            0.390196 |           |  0.275556  |      3.95576 |
    | qnn_bag_0.5 | ry_embedding        | HPzRx              |   0.333333 |            0.390196 |           |  0.275556  |      3.33663 |
    | qnn_bag_0.5 | rz_embedding        | tree_tensor        |   0.333333 |            0.390196 |           |  0.275556  |      5.62083 |
    | qnn_bag_0.5 | rz_embedding        | two_local          |   0.333333 |            0.390196 |           |  0.275556  |      4.47015 |
    | qnn_bag_0.5 | ZZ_embedding        | hardware_efficient |   0.333333 |            0.390196 |           |  0.275556  |     13.4961  |
    | qnn_bag_0.5 | ry_embedding        | tree_tensor        |   0.333333 |            0.390196 |           |  0.275556  |      5.50221 |
    | qnn_bag_0.5 | rx_embedding        | HPzRx              |   0.333333 |            0.390196 |           |  0.275556  |      3.32873 |
    | qnn_bag_0.5 | ry_embedding        | hardware_efficient |   0.333333 |            0.390196 |           |  0.275556  |     13.6979  |
    | qnn_bag_0.5 | rx_embedding        | two_local          |   0.333333 |            0.390196 |           |  0.275556  |      4.4354  |
    | qnn_bag_0.8 | rx_embedding        | tree_tensor        |   0.377778 |            0.38061  |           |  0.410005  |      5.84894 |
    | qnn_bag_0.8 | ZZ_embedding        | tree_tensor        |   0.377778 |            0.38061  |           |  0.410005  |      5.47199 |
    | qnn_bag_0.8 | rz_embedding        | tree_tensor        |   0.377778 |            0.38061  |           |  0.410005  |      5.97184 |
    | qnn_bag_0.8 | ry_embedding        | tree_tensor        |   0.377778 |            0.38061  |           |  0.410005  |      5.88461 |
    | qnn_bag_0.8 | ry_embedding        | two_local          |   0.355556 |            0.362092 |           |  0.3898    |      3.90912 |
    | qnn_bag_0.8 | rx_embedding        | two_local          |   0.355556 |            0.362092 |           |  0.3898    |      3.94093 |
    | qnn_bag_0.8 | rz_embedding        | two_local          |   0.355556 |            0.362092 |           |  0.3898    |      3.89387 |
    | qnn_bag_0.8 | ZZ_embedding        | two_local          |   0.355556 |            0.362092 |           |  0.3898    |      3.92849 |
    | qnn_bag_0.3 | ry_embedding        | HPzRx              |   0.222222 |            0.333333 |           |  0.0808081 |      3.34121 |
    | qnn_bag_0.3 | rx_embedding        | HPzRx              |   0.222222 |            0.333333 |           |  0.0808081 |      3.46206 |
    | qnn_bag_0.3 | rz_embedding        | HPzRx              |   0.222222 |            0.333333 |           |  0.0808081 |      3.39287 |
    | qnn_bag_0.3 | ZZ_embedding        | HPzRx              |   0.222222 |            0.333333 |           |  0.0808081 |      4.083   |
    | qnn_bag_0.3 | rx_embedding        | tree_tensor        |   0.222222 |            0.333333 |           |  0.0808081 |      5.86537 |
    | qnn_bag_0.3 | amplitude_embedding | two_local          |   0.222222 |            0.333333 |           |  0.0808081 |      4.51937 |
    | qnn_bag_0.3 | ZZ_embedding        | two_local          |   0.222222 |            0.333333 |           |  0.0808081 |      3.939   |
    | qnn_bag_0.3 | rz_embedding        | two_local          |   0.222222 |            0.333333 |           |  0.0808081 |      3.9069  |
    | qnn_bag_0.3 | ry_embedding        | two_local          |   0.222222 |            0.333333 |           |  0.0808081 |      3.88555 |
    | qnn_bag_0.3 | rx_embedding        | two_local          |   0.222222 |            0.333333 |           |  0.0808081 |      3.95736 |
    | qnn_bag_0.3 | amplitude_embedding | tree_tensor        |   0.222222 |            0.333333 |           |  0.0808081 |      5.45911 |
    | qnn_bag_0.3 | ZZ_embedding        | tree_tensor        |   0.222222 |            0.333333 |           |  0.0808081 |      5.95279 |
    | qnn_bag_0.3 | rz_embedding        | tree_tensor        |   0.222222 |            0.333333 |           |  0.0808081 |      6.21761 |
    | qnn_bag_0.3 | ry_embedding        | tree_tensor        |   0.222222 |            0.333333 |           |  0.0808081 |      5.92094 |
    | qnn_bag_0.3 | amplitude_embedding | hardware_efficient |   0.222222 |            0.333333 |           |  0.0808081 |     13.4813  |
    | qnn_bag_0.3 | ZZ_embedding        | hardware_efficient |   0.222222 |            0.333333 |           |  0.0808081 |     13.408   |
    | qnn_bag_0.3 | rz_embedding        | hardware_efficient |   0.222222 |            0.333333 |           |  0.0808081 |     13.3913  |
    | qnn_bag_0.3 | ry_embedding        | hardware_efficient |   0.222222 |            0.333333 |           |  0.0808081 |     13.354   |
    | qnn_bag_0.3 | rx_embedding        | hardware_efficient |   0.222222 |            0.333333 |           |  0.0808081 |     13.8513  |
    | qnn_bag_0.3 | amplitude_embedding | HPzRx              |   0.222222 |            0.333333 |           |  0.0808081 |      3.33983 |
    """

    def __init__(self, nqubits=8, randomstate=1234, predictions=False, ignoreWarnings=True, numPredictors=10, numLayers=5, customMetric=None, customImputerNum=None, customImputerCat=None, classifiers="all",verbose=1,optimizer=None,learningRate=0.1,epochs=100,runs=1,maxSamples=1.0):
        self.nqubits = nqubits
        self.randomstate = randomstate
        self.predictions = predictions
        self.ignoreWarnings = ignoreWarnings
        self.numLayers = numLayers
        self.numPredictors = numPredictors
        self.customMetric = customMetric
        self.classifiers = classifiers
        self.verbose = verbose
        self.learninRate = learningRate
        self.epochs = epochs
        self.runs = runs
        self.maxSamples = maxSamples

        self.verboseprint = print if verbose else lambda *a, **k: None

        if optimizer is None:
            self.verboseprint("No optimizer has been passed adam will be used by default.")
            self.optimizer = optax.adam(learning_rate=self.learninRate)
        else:
            if isinstance(optimizer, optax.GradientTransformation):
                self.verboseprint("Optimizer is an optax optimizer; setting its learning rate.")
                self.optimizer = optimizer.replace(learning_rate=self.learningRate)
            else:
                self.verboseprint("Optimizer is not from optax library; using the provided optimizer.")
                self.optimizer = optimizer
        
        if customImputerNum is not None:
            module = inspect.getmodule(customImputerNum)
            # Check if the module belongs to sklearn.impute
            if module.__name__.startswith('sklearn.impute'):
                self.verboseprint("The object belongs to the sklearn.impute module. Custom Numeric Imputer will be used.")
                self.numeric_transformer = Pipeline(
                steps=[("imputer",customImputerNum), ("scaler", StandardScaler())])
            else:
                self.verboseprint("The object does not belong to the sklearn.impute module. Default Custom Numeric Imputer will be used.")
        else:
            self.numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())])


        if customImputerCat is not None:
            module = inspect.getmodule(customImputerNum)
            # Check if the module belongs to sklearn.impute
            if module.__name__.startswith('sklearn.impute'):
                self.verboseprint("The object belongs to the sklearn.impute module. Custom Numeric Categorical will be used.")
                self.categorical_transformer = Pipeline(
                steps=[("imputer",customImputerCat), ("scaler", StandardScaler())])
            else:
                self.verboseprint("The object does not belong to the sklearn.impute module. Default Custom Categorical Imputer will be used.")
        else:    
            self.categorical_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())])


    
    def fit(self, X_train, X_test, y_train, y_test):
        """
        Fit Classification algorithms to X_train and y_train, predict and score on X_test, y_test.
        If the dimensions of the training vectors are not compatible with the different models, a 
        PCA transformation will be used in order to reduce the dimensionality to a compatible space.
        All categories must be in the training data if there are new categories in the test date the
        function will not work. The objective variable must be in a numerical form like LabelEncoder or
        OrdinalEncoder. Onehot or strings won't work.

        Parameters
        ----------
        X_train : array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.
        X_test : array-like,
            Testing vectors, where rows is the number of samples
            and columns is the number of features.
        y_train : array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.
        y_test : array-like,
            Testing vectors, where rows is the number of samples
            and columns is the number of features.
        
        Returns
        -------
        scores : Pandas DataFrame
            Returns metrics of all the models in a Pandas DataFrame.
        predictions : Pandas DataFrame
            Returns predictions of all the models in a Pandas DataFrame.
        """
        # TABLE
        NAMES = []
        EMBEDDINGS = []
        ANSATZ = []
        ACCURACY = []
        B_ACCURACY = []
        ROC_AUC = []
        F1 = []
        TIME = []
        PARAMETERS = []

        predictions = {}
        
        models = []
        

        numClasses = len(np.unique(y_train))
        binary = numClasses == 2 


        if self.customMetric is not None:
            customMetric = []

        if isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(X_train)
            X_test = pd.DataFrame(X_test)

        y_test = y_test[~np.isnan(y_test)]
        y_train = y_train[~np.isnan(y_train)]

        numeric_features = X_train.select_dtypes(include=[np.number]).columns
        categorical_features = X_train.select_dtypes(include=["object"]).columns

        
        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", self.numeric_transformer, numeric_features),
                ("categorical_low", self.categorical_transformer, categorical_features),
            ]
        )

        X_train=preprocessor.fit_transform(X_train)
        X_test=preprocessor.transform(X_test)

        pca = PCA(n_components=self.nqubits)
        pca_amp = PCA(n_components=2**self.nqubits)

        X_train_amp = pca_amp.fit_transform(X_train) if 2**self.nqubits <= X_train.shape[1] else X_train
        X_test_amp = pca_amp.transform(X_test) if 2**self.nqubits <= X_test.shape[1] else X_test

        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        
        
        one = OneHotEncoder(sparse_output=False)

        # Creates tuple of (model_name, embedding, ansatz, features)
        if self.classifiers == "all":
            [models.append(("qsvm", embedding, None, None)) for embedding in embeddings]
            [models.append(("qnn", embedding, ansatz, None)) for ansatz in ansatzs for embedding in embeddings]
            [models.append(("qnn_bag", embedding, ansatz, feature)) for ansatz in ansatzs for embedding in embeddings for feature in features]
        else:
            if "qsvm" in self.classifiers:
                [models.append(("qsvm", embedding, None, None)) for embedding in embeddings]
            if "qnn" in self.classifiers :
                [models.append(("qnn", embedding, ansatz, None)) for ansatz in ansatzs for embedding in embeddings]
            if "qnn_bag" in self.classifiers:
                [models.append(("qnn_bag", embedding, ansatz, feature)) for ansatz in ansatzs for embedding in embeddings for feature in features]
            if models == []:
                raise Exception("Error no valid classifiers selected. It should belong to the following list [\"all\",\"qsvm\",\"qnn\",\"qnn_bag\"]")

        for model in models:
            
            name, embedding, ansatz, feature = model
            name = name if feature==None else name + f"_{feature}"
            self.verboseprint(name,embedding,ansatz)

            if name == "qsvm":        
                start = time.time()
                qsvm = SVC(kernel= qkernel(embedding, n_qubits=self.nqubits))
                qsvm.fit(X=X_train if embedding != "amplitude_embedding" else X_train_amp,y=y_train)
                y_pred = qsvm.predict(X=X_test if embedding != "amplitude_embedding" else X_test_amp)
                accuracy = accuracy_score(y_test, y_pred, normalize=True)
                b_accuracy = balanced_accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="weighted")
                try:
                    roc_auc = roc_auc_score(y_test, y_pred)
                except Exception as exception:
                    roc_auc = None
                    if self.ignoreWarnings is False:
                        self.verboseprint("ROC AUC couldn't be calculated for " + name)
                        self.verboseprint(exception)
               
            elif name == "qnn":
                if binary:
                    start = time.time()
                    qnn_tmp = create_circuit_binary(self.nqubits, layers=self.numLayers, ansatz=ansatz)
                    # apply vmap on x (first circuit param)
                    qnn_batched = jax.vmap(qnn_tmp, (0, None))
                    # Jit for faster execution
                    qnn = jax.jit(qnn_batched)
                    preds, accuracy, b_accuracy, f1, roc_auc = evaluate_full_model_predictor_binary(qnn=qnn,optimizer=self.optimizer,epochs=self.epochs,n_qubits=self.nqubits,layers=self.numLayers,ansatz=ansatz,X_train=X_train if embedding != "amplitude_embedding" else X_train_amp,X_test=X_test if embedding != "amplitude_embedding" else X_test_amp,y_train=y_train,y_test=y_test,runs=self.runs,seed=self.randomstate,verboseprint=self.verboseprint)
                    if self.predictions:
                        predictions.append(preds)
                else:
                    y_train_o = one.fit_transform(y_train.reshape(-1,1))
                    y_test_o = one.transform(y_test.reshape(-1,1))
                    start = time.time()
                    qnn_tmp = create_circuit(self.nqubits, layers=self.numLayers, ansatz=ansatz, n_class=numClasses)
                    # apply vmap on x (first circuit param)
                    qnn_batched = jax.vmap(qnn_tmp, (0, None))
                    # Jit for faster execution
                    qnn = jax.jit(qnn_batched)
                    preds, accuracy, b_accuracy, f1, roc_auc = evaluate_full_model_predictor(qnn=qnn,optimizer=self.optimizer,epochs=self.epochs,n_qubits=self.nqubits,layers=self.numLayers,ansatz=ansatz,X_train=X_train if embedding != "amplitude_embedding" else X_train_amp,X_test=X_test if embedding != "amplitude_embedding" else X_test_amp,y_train=y_train_o,y_test=y_test_o,seed=self.randomstate,runs=self.runs,verboseprint=self.verboseprint)
                    if self.predictions:
                        predictions.append(preds)
            elif "qnn_bag" in name:
                if binary:
                    start = time.time()
                    qnn_tmp_bag = create_circuit_binary(n_qubits=self.nqubits,layers=self.numLayers,ansatz=ansatz)
                    # apply vmap on x (first circuit param)
                    qnn_batched_bag = jax.vmap(qnn_tmp_bag, (0, None))
                    # Jit for faster execution
                    qnn_bag = jax.jit(qnn_batched_bag)
                    preds, accuracy, b_accuracy, f1, roc_auc = evaluate_bagging_predictor_binary(qnn=qnn_bag,optimizer=self.optimizer,epochs=self.epochs,n_qubits=self.nqubits,layers=self.numLayers,ansatz=ansatz,X_train=X_train if embedding != "amplitude_embedding" else X_train_amp,X_test=X_test if embedding != "amplitude_embedding" else X_test_amp,y_train=y_train,y_test=y_test,seed=self.randomstate,runs=self.runs,n_estimators=self.numPredictors,max_features=feature,max_samples=self.maxSamples,verboseprint=self.verboseprint)
                    if self.predictions:
                        predictions.append(preds)
                else:
                    start = time.time()
                    y_train_o =one.fit_transform(y_train.reshape(-1,1))
                    y_test_o = one.transform(y_test.reshape(-1,1))
                    qnn_tmp_bag = create_circuit(n_qubits=self.nqubits,layers=self.numLayers,ansatz=ansatz,n_class=numClasses)
                    # apply vmap on x (first circuit param)
                    qnn_batched_bag = jax.vmap(qnn_tmp_bag, (0, None))
                    # Jit for faster execution
                    qnn_bag = jax.jit(qnn_batched_bag)
                    preds, accuracy, b_accuracy, f1, roc_auc = evaluate_bagging_predictor(qnn=qnn_bag,optimizer=self.optimizer,epochs=self.epochs,n_qubits=self.nqubits,layers=self.numLayers,ansatz=ansatz,X_train=X_train if embedding != "amplitude_embedding" else X_train_amp,X_test=X_test if embedding != "amplitude_embedding" else X_test_amp,y_train=y_train_o,y_test=y_test_o,seed=self.randomstate,runs=self.runs,n_estimators=self.numPredictors,max_features=feature,max_samples=self.maxSamples,verboseprint=self.verboseprint)
                    if self.predictions:
                        predictions.append(preds)

            NAMES.append(name)
            ANSATZ.append(ansatz)
            ACCURACY.append(accuracy)
            B_ACCURACY.append(b_accuracy)
            ROC_AUC.append(roc_auc)
            EMBEDDINGS.append(embedding)
            F1.append(f1)
            TIME.append(time.time() - start)

            if self.customMetric is not None:
                customMetric = self.customMetric(y_test, y_pred)
                customMetric.append(customMetric)
            if self.verbose > 0:
                if self.customMetric is not None:
                    self.verboseprint(
                        {
                            "Model": NAMES[-1],
                            "Embedding": EMBEDDINGS[-1],
                            "Ansatz": ANSATZ[-1],
                            "Accuracy": ACCURACY[-1],
                            "Balanced Accuracy": B_ACCURACY[-1],
                            "ROC AUC": ROC_AUC[-1],
                            "F1 Score": F1[-1],
                            self.customMetric.__name__: customMetric,
                            "Time taken": TIME[-1],
                        }
                    )
                else:
                    self.verboseprint(
                        {
                            "Model": NAMES[-1],
                            "Embedding": EMBEDDINGS[-1],
                            "Ansatz": ANSATZ[-1],
                            "Accuracy": ACCURACY[-1],
                            "Balanced Accuracy": B_ACCURACY[-1],
                            "ROC AUC": ROC_AUC[-1],
                            "F1 Score": F1[-1],
                            "Time taken": TIME[-1],
                        }
                    )
            
            
        if self.customMetric is None:
           
            scores = pd.DataFrame(
                {
                    "Model": NAMES,
                    "Embedding": EMBEDDINGS,
                    "Ansatz": ANSATZ,
                    "Accuracy": ACCURACY,
                    "Balanced Accuracy": B_ACCURACY,
                    "ROC AUC": ROC_AUC,
                    "F1 Score": F1,
                    "Time taken": TIME,
                }
            )
        else:
            scores = pd.DataFrame(
                {
                    "Model": NAMES,
                    "Embedding": EMBEDDINGS,
                    "Ansatz": ANSATZ,
                    "Accuracy": ACCURACY,
                    "Balanced Accuracy": B_ACCURACY,
                    "ROC AUC": ROC_AUC,
                    "F1 Score": F1,
                    self.customMetric.__name__: customMetric,
                    "Time taken": TIME,
                }
            )
        scores = scores.sort_values(by="Balanced Accuracy", ascending=False).set_index(
            "Model"
        )
        if self.predictions:
            predictions_df = pd.DataFrame.from_dict(predictions)
        
        print(scores.to_markdown())
        return scores, predictions_df if self.predictions is True else scores

from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split

data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.3,random_state =123)  

q = QuantumClassifier(nqubits=4,classifiers="all",verbose=0)

scores, predicitons = q.fit(X_train, X_test, y_train, y_test)

print(scores.to_markdown()) 




