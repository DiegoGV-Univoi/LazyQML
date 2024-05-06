#======================================================================================
# Main Module
#======================================================================================

"""
 Import modules
"""

from common import *

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
        For the liblinear and lbfgs solvers set verbose to any positive
        number for verbosity.
    ignoreWarnings : bool, optional (default=True)
        When set to True, the warning related to algorigms that are not able to run are ignored.
    customMetric : function, optional (default=None)
        When function is provided, models are evaluated based on the custom evaluation metric provided.
    customImputerNum : function, optional (default=None)
        When function is provided, models are imputed based on the custom numeric imputer provided.
    customImputerCat : function, optional (default=None)
        When function is provided, models are imputed based on the custom categorical imputer provided.
    prediction : bool, optional (default=False)
        When set to True, the predictions of all the models models are returned as dataframe.
    classifiers : string, optional (default="all")
        When function is provided, trains the chosen classifier(s) ["all", "qsvm", "qnn", "qnnbag"].

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
    """

    def __init__(self, nqubits=8, randomstate=1234, predictions=False, ignoreWarnings=True, numPredictors=10, numLayers=5, customMetric=None, customImputerNum=None, customImputerCat=None, classifiers="all",verbose=1,optimizer=None,learningRate=0.1,epochs=100,runs=1,maxsamples=1):
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
        self.maxsamples = maxsamples

        if optimizer is None:
            self.optimizer = optax.adam(learning_rate=self.learninRate)
        else:
            self.optimizer = optimizer
        
        if customImputerNum is not None:
            module = inspect.getmodule(customImputerNum)
            # Check if the module belongs to sklearn.impute
            if module.__name__.startswith('sklearn.impute'):
                print("The object belongs to the sklearn.impute module. Custom Numeric Imputer will be used.")
                self.numeric_transformer = Pipeline(
                steps=[("imputer",customImputerNum), ("scaler", StandardScaler())])
            else:
                print("The object does not belong to the sklearn.impute module. Default Custom Numeric Imputer will be used.")
        else:
            self.numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())])


        if customImputerCat is not None:
            module = inspect.getmodule(customImputerNum)
            # Check if the module belongs to sklearn.impute
            if module.__name__.startswith('sklearn.impute'):
                print("The object belongs to the sklearn.impute module. Custom Numeric Categorical will be used.")
                self.categorical_transformer = Pipeline(
                steps=[("imputer",customImputerCat), ("scaler", StandardScaler())])
            else:
                print("The object does not belong to the sklearn.impute module. Default Custom Categorical Imputer will be used.")
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
            print(name,embedding,ansatz)

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
                        print("ROC AUC couldn't be calculated for " + name)
                        print(exception)
               
            elif name == "qnn":
                if binary:
                    start = time.time()
                    qnn_tmp = create_circuit_binary(self.nqubits, layers=self.numLayers, ansatz=ansatz)
                    # apply vmap on x (first circuit param)
                    qnn_batched = jax.vmap(qnn_tmp, (0, None))
                    # Jit for faster execution
                    qnn = jax.jit(qnn_batched)
                    preds, accuracy, b_accuracy, f1, roc_auc = evaluate_full_model_predictor_binary(qnn=qnn,optimizer=self.optimizer,epochs=self.epochs,n_qubits=self.nqubits,layers=self.numLayers,ansatz=ansatz,X_train=X_train if embedding != "amplitude_embedding" else X_train_amp,X_test=X_test if embedding != "amplitude_embedding" else X_test_amp,y_train=y_train,y_test=y_test,runs=self.runs,seed=self.randomstate)
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
                    preds, accuracy, b_accuracy, f1, roc_auc = evaluate_full_model_predictor(qnn=qnn,optimizer=self.optimizer,epochs=self.epochs,n_qubits=self.nqubits,layers=self.numLayers,ansatz=ansatz,X_train=X_train if embedding != "amplitude_embedding" else X_train_amp,X_test=X_test if embedding != "amplitude_embedding" else X_test_amp,y_train=y_train_o,y_test=y_test_o,seed=self.randomstate,runs=self.runs)
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
                    preds, accuracy, b_accuracy, f1, roc_auc = evaluate_bagging_predictor_binary(qnn=qnn_bag,optimizer=self.optimizer,epochs=self.epochs,n_qubits=self.nqubits,layers=self.numLayers,ansatz=ansatz,X_train=X_train if embedding != "amplitude_embedding" else X_train_amp,X_test=X_test if embedding != "amplitude_embedding" else X_test_amp,y_train=y_train,y_test=y_test,seed=self.randomstate,runs=self.runs,n_estimators=self.numPredictors,max_features=feature,max_samples=self.maxsamples)
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
                    preds, accuracy, b_accuracy, f1, roc_auc = evaluate_bagging_predictor(qnn=qnn_bag,optimizer=self.optimizer,epochs=self.epochs,n_qubits=self.nqubits,layers=self.numLayers,ansatz=ansatz,X_train=X_train if embedding != "amplitude_embedding" else X_train_amp,X_test=X_test if embedding != "amplitude_embedding" else X_test_amp,y_train=y_train_o,y_test=y_test_o,seed=self.randomstate,runs=self.runs,n_estimators=self.numPredictors,max_features=feature,max_samples=self.maxsamples)
                    if self.predictions:
                        predictions.append(preds)

            NAMES.append(name)
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
                    print(
                        {
                            "Model": NAMES[-1],
                            "Embedding": EMBEDDINGS[-1],
                            "Accuracy": ACCURACY[-1],
                            "Balanced Accuracy": B_ACCURACY[-1],
                            "ROC AUC": ROC_AUC[-1],
                            "F1 Score": F1[-1],
                            self.customMetric.__name__: customMetric,
                            "Time taken": TIME[-1],
                        }
                    )
                else:
                    print(
                        {
                            "Model": NAMES[-1],
                            "Embedding": EMBEDDINGS[-1],
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
        return scores, predictions_df if self.predictions is True else scores


from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
data = load_iris()
X = data.data
y = data.target
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.3,random_state =123)  

q = QuantumClassifier(nqubits=4,classifiers=["qnn_bag"],verbose=0)

scores = q.fit(X_train, X_test, y_train, y_test)

print(scores)



