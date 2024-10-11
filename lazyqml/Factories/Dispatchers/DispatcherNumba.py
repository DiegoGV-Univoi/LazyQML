from numba import jit, prange
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import time
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
from Factories.Models.fModels import *
from Factories.Preprocessing.fPreprocessing import *
from Utils.Utils import *
import time 
import math

class Dispatcher:
    def __init__(self, sequential, threshold=27):
        self.sequential = sequential
        self.threshold = threshold

    @staticmethod
    @jit(nopython=True, parallel=True)
    def _parallel_predictions(model_predict_func, X_test_values):
        """
        Parallel prediction using Numba
        Note: This is a simplified version - actual implementation would need 
        to account for model specifics
        """
        n_samples = X_test_values.shape[0]
        predictions = np.zeros(n_samples)
        
        for i in prange(n_samples):
            predictions[i] = model_predict_func(X_test_values[i])
        
        return predictions

    @staticmethod
    @jit(nopython=True)
    def _calculate_metrics(y_true, y_pred):
        """
        Calculate basic metrics using Numba
        Note: This is a simplified version - actual implementation would need
        to match sklearn's metric calculations
        """
        n_samples = len(y_true)
        correct = 0
        for i in range(n_samples):
            if y_true[i] == y_pred[i]:
                correct += 1
        return correct / n_samples

    def _executeModel(self, model, X_train, y_train, X_test, y_test, predictions, runs, customMetric):
        preds = []
        accuracyR, b_accuracyR, f1R, customR = 0, 0, 0, 0
        custom = None
        total_exeT = 0

        # Convert data to numpy arrays for Numba compatibility
        X_train_values = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        X_test_values = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
        y_train_values = y_train.values if isinstance(y_train, pd.Series) else y_train
        y_test_values = y_test.values if isinstance(y_test, pd.Series) else y_test

        for j in range(runs):
            printer.print(f"\tExecuting {j+1} run of {runs}")
            start = time.time()
            
            # Training can't be easily Numba-optimized due to model complexity
            model.fit(X=X_train, y=y_train)
            
            # Try to use Numba for predictions if possible
            try:
                y_pred = self._parallel_predictions(model.predict, X_test_values)
            except:
                # Fallback to regular prediction if Numba fails
                y_pred = model.predict(X=X_test)
            
            run_time = time.time() - start
            total_exeT += run_time

            # Try to use Numba for metric calculation
            try:
                accuracy = self._calculate_metrics(y_test_values, y_pred)
                accuracyR += accuracy
            except:
                # Fallback to sklearn metrics if Numba fails
                accuracyR += accuracy_score(y_test, y_pred, normalize=True)
            
            # These metrics are more complex and may not benefit from Numba
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

        avg_exeT = total_exeT/runs
        return avg_exeT, accuracy, b_accuracy, f1, custom, preds

    def dispatch(self, nqubits, randomstate, predictions, shots,
                numPredictors, numLayers, classifiers, ansatzs, backend,
                embeddings, features, learningRate, epochs, runs, batch,
                maxSamples, verbose, customMetric, customImputerNum, customImputerCat,
                X_train, y_train, X_test, y_test, showTable=True):
        
        NAMES, EMBEDDINGS, ANSATZ, ACCURACY = [], [], [], []
        B_ACCURACY, FEATURES, F1, TIME, CUSTOM = [], [], [], [], []
        
        numClasses = len(np.unique(y_train))
        
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
        
        models_to_run = []
        for combination in combinations:
            name, embedding, ansatz, feature = combination
            printer.print("="*100)
            feature = feature if feature is not None else "~"
            printer.print(f"Model: {name} Embedding: {embedding} Ansatz:{ansatz} Features: {feature}")
            
            model = ModelFactory().getModel(
                Nqubits=adjustedQubits, model=name, Embedding=embedding,
                Ansatz=ansatz, N_class=numClasses, backend=backend,
                Shots=shots, seed=randomstate, Layers=numLayers,
                Max_samples=maxSamples, Max_features=feature,
                LearningRate=learningRate, BatchSize=batch,
                Epoch=epochs, numPredictors=numPredictors
            )
            
            preprocessing = prepFactory.GetPreprocessing(ansatz=ansatz, embedding=embedding)
            X_train_processed = preprocessing.fit_transform(X_train, y=y_train)
            X_test_processed = preprocessing.transform(X_test)
            
            models_to_run.append((
                model, X_train_processed, y_train, X_test_processed, y_test,
                predictions, runs, customMetric, name, embedding, ansatz, feature
            ))

        if self.sequential or backend == "Lightning.GPU" or nqubits >= self.threshold:
            # Sequential execution
            for model_args in models_to_run:
                self._process_single_model(model_args, NAMES, EMBEDDINGS, ANSATZ, ACCURACY,
                                          B_ACCURACY, FEATURES, F1, TIME, CUSTOM)
        else:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=min(len(models_to_run), psutil.cpu_count())) as executor:
                futures = []
                for model_args in models_to_run:
                    future = executor.submit(self._process_single_model, model_args,
                                            NAMES, EMBEDDINGS, ANSATZ, ACCURACY,
                                            B_ACCURACY, FEATURES, F1, TIME, CUSTOM)
                    futures.append(future)
                
                for future in futures:
                    future.result()

        return self._create_results_dataframe(NAMES, EMBEDDINGS, ANSATZ, FEATURES,
                                             ACCURACY, B_ACCURACY, F1, CUSTOM, TIME,
                                             customMetric, showTable)

    def _process_single_model(self, model_args, NAMES, EMBEDDINGS, ANSATZ, ACCURACY,
                             B_ACCURACY, FEATURES, F1, TIME, CUSTOM):
        model, X_train_p, y_train, X_test_p, y_test, preds, runs, custom_metric, \
        name, embedding, ansatz, feature = model_args
        
        exeT, accuracy, b_accuracy, f1, custom, _ = self._executeModel(
            model, X_train_p, y_train, X_test_p, y_test, preds, runs, custom_metric
        )
        
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
        columns = {
            "Model": NAMES,
            "Embedding": EMBEDDINGS,
            "Ansatz": ANSATZ,
            "Features": FEATURES,
            "Accuracy": ACCURACY,
            "Balanced Accuracy": B_ACCURACY,
            "F1 Score": F1,
            "Time taken": TIME,
        }
        
        if customMetric is not None:
            columns[customMetric.__name__] = CUSTOM
        
        scores = pd.DataFrame(columns)
        
        if showTable:
            print(scores.to_markdown())
        
        return scores