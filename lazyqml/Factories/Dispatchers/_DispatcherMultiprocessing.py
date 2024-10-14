from multiprocessing import Pool
from Utils.Utils import * 
import numpy as np
import pandas as pd
import math
from Factories.Models.fModels import *
from Factories.Preprocessing.fPreprocessing import *
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
import time
import psutil

class Dispatcher:
    def __init__(self, sequential, threshold=27):
        self.sequential = sequential
        self.threshold = threshold
        
    def _executeModel(self, args):
        model_params, X_train, y_train, X_test, y_test, predictions, runs, customMetric = args
        
        # Recreate the model in this process
        model = ModelFactory().getModel(**model_params)
        
        preds = []
        accuracyR, b_accuracyR, f1R, customR = 0, 0, 0, 0
        custom = None
        exeT = 0
        
        for j in range(runs):
            start = time.time()
            model.fit(X=X_train, y=y_train)
            exeT += time.time() - start
            y_pred = model.predict(X=X_test)
            
            accuracyR += accuracy_score(y_test, y_pred, normalize=True)
            b_accuracyR += balanced_accuracy_score(y_test, y_pred)
            f1R += f1_score(y_test, y_pred, average="weighted")
            if customMetric is not None:
                customR += customMetric(y_test, y_pred)
        
        if predictions:
            preds = y_pred
        accuracy = accuracyR/runs
        b_accuracy = b_accuracyR/runs
        f1 = f1R/runs
        if customMetric is not None:
            custom = customR/runs
        
        exeT = exeT/runs
        return exeT, accuracy, b_accuracy, f1, custom, preds

    def dispatch(self, nqubits, randomstate, predictions, shots,
                numPredictors, numLayers, classifiers, ansatzs, backend,
                embeddings, features, learningRate, epochs, runs, batch,
                maxSamples, verbose, customMetric, customImputerNum, customImputerCat,
                X_train, y_train, X_test, y_test, showTable=True):
        
        NAMES, EMBEDDINGS, ANSATZ, ACCURACY = [], [], [], []
        B_ACCURACY, FEATURES, F1, TIME, CUSTOM = [], [], [], [], []
        
        numClasses = len(np.unique(y_train))
        
        # Adjust qubits if necessary
        if (numClasses > 2**math.floor(math.log2(nqubits))):
            printer.print("The number of qubits must exceed the number of classes and be a power of 2.")
            adjustedQubits = adjustQubits(nqubits=nqubits, numClasses=numClasses)
            printer.print(f"New number of qubits:\t{adjustedQubits}")
        else:
            adjustedQubits = nqubits
        
        # Convert input data to pandas DataFrames if needed
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train)
        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test)
        
        # Preprocessing
        prepFactory = PreprocessingFactory(nqubits)
        sanitizer = prepFactory.GetSanitizer(customImputerCat, customImputerNum)
        X_train = sanitizer.fit_transform(X_train)
        X_test = sanitizer.transform(X_test)
        
        combinations = create_combinations(classifiers=classifiers, embeddings=embeddings, 
                                          features=features, ansatzs=ansatzs)
        
        execution_args = []
        for combination in combinations:
            name, embedding, ansatz, feature = combination
            printer.print("="*100)
            feature = feature if feature is not None else "~"
            printer.print(f"Model: {name} Embedding: {embedding} Ansatz:{ansatz} Features: {feature}")
            
            # Instead of creating the model, we'll just store its parameters
            model_params = {
                'Nqubits': adjustedQubits,
                'model': name,
                'Embedding': embedding,
                'Ansatz': ansatz,
                'N_class': numClasses,
                'backend': backend,
                'Shots': shots,
                'seed': randomstate,
                'Layers': numLayers,
                'Max_samples': maxSamples,
                'Max_features': feature,
                'LearningRate': learningRate,
                'BatchSize': batch,
                'Epoch': epochs,
                'numPredictors': numPredictors
            }
            
            preprocessing = prepFactory.GetPreprocessing(ansatz=ansatz, embedding=embedding)
            X_train_processed = preprocessing.fit_transform(X_train, y=y_train)
            X_test_processed = preprocessing.transform(X_test)
            
            execution_args.append((
                model_params, X_train_processed, y_train, X_test_processed, y_test,
                predictions, runs, customMetric, name, embedding, ansatz, feature
            ))

        if self.sequential or backend == "Lightning.GPU" or nqubits >= self.threshold:
            # Sequential execution
            for args in execution_args:
                model_params, X_train_p, y_train, X_test_p, y_test, preds, runs, \
                custom_metric, name, embedding, ansatz, feature = args
                
                exe_args = (model_params, X_train_p, y_train, X_test_p, y_test, 
                           preds, runs, custom_metric)
                exeT, accuracy, b_accuracy, f1, custom, _ = self._executeModel(exe_args)
                
                self._append_results(NAMES, EMBEDDINGS, ANSATZ, ACCURACY, B_ACCURACY,
                                    FEATURES, F1, TIME, CUSTOM, name, embedding, ansatz,
                                    feature, accuracy, b_accuracy, f1, exeT, custom)
        else:
            # Parallel execution using multiprocessing
            num_processes = min(len(execution_args), psutil.cpu_count(logical=False))
            with Pool(processes=num_processes) as pool:
                results = []
                for args in execution_args:
                    model_params, X_train_p, y_train, X_test_p, y_test, preds, runs, \
                    custom_metric, name, embedding, ansatz, feature = args
                    
                    exe_args = (model_params, X_train_p, y_train, X_test_p, y_test, 
                               preds, runs, custom_metric)
                    result = pool.apply_async(self._executeModel, (exe_args,))
                    results.append((result, name, embedding, ansatz, feature))
                
                for result, name, embedding, ansatz, feature in results:
                    exeT, accuracy, b_accuracy, f1, custom, _ = result.get()
                    self._append_results(NAMES, EMBEDDINGS, ANSATZ, ACCURACY, B_ACCURACY,
                                        FEATURES, F1, TIME, CUSTOM, name, embedding, ansatz,
                                        feature, accuracy, b_accuracy, f1, exeT, custom)
        
        return self._create_results_dataframe(NAMES, EMBEDDINGS, ANSATZ, FEATURES,
                                             ACCURACY, B_ACCURACY, F1, CUSTOM, TIME,
                                             customMetric, showTable)

    # _append_results and _create_results_dataframe methods remain unchanged
    def _append_results(self, NAMES, EMBEDDINGS, ANSATZ, ACCURACY, B_ACCURACY,
                        FEATURES, F1, TIME, CUSTOM, name, embedding, ansatz,
                        feature, accuracy, b_accuracy, f1, exeT, custom):
        NAMES.append(name)
        EMBEDDINGS.append(embedding)
        ANSATZ.append(ansatz)
        ACCURACY.append(accuracy)
        B_ACCURACY.append(b_accuracy)
        FEATURES.append(feature)
        F1.append(f1)
        TIME.append(exeT)
        CUSTOM.append(custom)

    def _create_results_dataframe(self, NAMES, EMBEDDINGS, ANSATZ, FEATURES,
                                 ACCURACY, B_ACCURACY, F1, CUSTOM, TIME,
                                 customMetric, showTable):
        if customMetric is None:
            scores = pd.DataFrame({
                "Model": NAMES,
                "Embedding": EMBEDDINGS,
                "Ansatz": ANSATZ,
                "Features": FEATURES,
                "Accuracy": ACCURACY,
                "Balanced Accuracy": B_ACCURACY,
                "F1 Score": F1,
                "Time taken": TIME,
            })
        else:
            scores = pd.DataFrame({
                "Model": NAMES,
                "Embedding": EMBEDDINGS,
                "Ansatz": ANSATZ,
                "Features": FEATURES,
                "Accuracy": ACCURACY,
                "Balanced Accuracy": B_ACCURACY,
                "F1 Score": F1,
                customMetric.__name__: CUSTOM,
                "Time taken": TIME,
            })
        
        if showTable:
            print(scores.to_markdown())
        
        return scores