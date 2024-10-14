from Utils.Utils import *
import numpy as np
import pandas as pd
import math
from Factories.Models.fModels import *
from Factories.Preprocessing.fPreprocessing import *
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
import time
from joblib import Parallel, delayed

class Dispatcher:
    def __init__(self, sequential=False, threshold=27):
        self.sequential = sequential
        self.threshold = threshold

    def execute_model(self, model_factory_params, X_train, y_train, X_test, y_test, predictions, runs, customMetric):
        model = ModelFactory().getModel(**model_factory_params)
        preds = []
        accuracyR, b_accuracyR, f1R, customR = 0, 0, 0, 0
        custom = None
    
        
        start = time.time()
        for _ in range(runs):
            model.fit(X=X_train, y=y_train)
            y_pred = model.predict(X=X_test)

            accuracyR += accuracy_score(y_test, y_pred, normalize=True)
            b_accuracyR += balanced_accuracy_score(y_test, y_pred)
            f1R += f1_score(y_test, y_pred, average="weighted")
            if customMetric is not None:
                customR += customMetric(y_test, y_pred)

        exeT = time.time() - start

        if predictions:
            preds = y_pred
        accuracy = accuracyR / runs
        b_accuracy = b_accuracyR / runs
        f1 = f1R / runs
        if customMetric is not None:
            custom = customR / runs

        return model_factory_params['model'], model_factory_params['Embedding'], model_factory_params['Ansatz'], model_factory_params['Max_features'], exeT, accuracy, b_accuracy, f1, custom, preds

    def dispatch(self, nqubits, randomstate, predictions, shots, 
                 numPredictors, numLayers, classifiers, ansatzs, backend, 
                 embeddings, features, learningRate, epochs, runs, batch,
                 maxSamples, verbose, customMetric, customImputerNum, 
                 customImputerCat, X_train, y_train, X_test, y_test, 
                 showTable=True):
        
        combinations = create_combinations(classifiers=classifiers, 
                                           embeddings=embeddings, 
                                           features=features, 
                                           ansatzs=ansatzs)
        
        
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train)
        if not isinstance(X_test, pd.DataFrame):
                X_test = pd.DataFrame(X_test)

        numClasses = len(np.unique(y_train))
        adjustedQubits = adjustQubits(nqubits=nqubits, numClasses=numClasses)

        prepFactory = PreprocessingFactory(adjustedQubits)
        sanitizer = prepFactory.GetSanitizer(customImputerCat, customImputerNum)

        X_train = sanitizer.fit_transform(X_train)
        X_test = sanitizer.transform(X_test)

        # Calculate available memory and determine max parallel models
        available_memory = calculate_free_memory()
        quantum_memory = calculate_quantum_memory(num_qubits=adjustedQubits)
        max_models_parallel = min(int(available_memory // quantum_memory) if quantum_memory > 0 else float('inf'), psutil.cpu_count(logical=False))

        print(f"MAX MODELS: {max_models_parallel}")

        
        # Prepare all model executions
        all_executions = []
        for combination in combinations:
            name, embedding, ansatz, feature = combination
            feature = feature if feature is not None else "~"
            
            model_factory_params = {
                "Nqubits": adjustedQubits,
                "model": name,
                "Embedding": embedding,
                "Ansatz": ansatz,
                "N_class": numClasses,
                "backend": backend,
                "Shots": shots,
                "seed": randomstate,
                "Layers": numLayers,
                "Max_samples": maxSamples,
                "Max_features": feature,
                "LearningRate": learningRate,
                "BatchSize": batch,
                "Epoch": epochs,
                "numPredictors": numPredictors
            }

            preprocessing = prepFactory.GetPreprocessing(ansatz=ansatz, embedding=embedding)
            X_train_processed = preprocessing.fit_transform(X_train, y=y_train)
            X_test_processed = preprocessing.transform(X_test)

            all_executions.append((model_factory_params, X_train_processed, y_train, X_test_processed, y_test, predictions, runs, customMetric))
 
        # Execute all models in parallel
        if self.sequential:
            results = [self.execute_model(*execution_params) for execution_params in all_executions]
        else:
            results = Parallel(n_jobs=max_models_parallel, prefer='processes')(
                delayed(self.execute_model)(*execution_params) for execution_params in all_executions
            )

        # Process results
        scores = pd.DataFrame(results, columns=["Model", "Embedding", "Ansatz", "Features", "Time taken", "Accuracy", "Balanced Accuracy", "F1 Score", "Custom Metric", "Predictions"])

        if showTable:
            print(scores.to_markdown())

        return scores