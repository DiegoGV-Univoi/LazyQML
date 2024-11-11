from Utils.Utils import *
import numpy as np
import pandas as pd
import math
from Factories.Models.fModels import *
from Factories.Preprocessing.fPreprocessing import *
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
import time
from joblib import Parallel, delayed
from sklearn.model_selection import RepeatedKFold

class DispatcherCV:
    def __init__(self, sequential=False, threshold=20, time=True, folds=10, repeats=5):
        self.sequential = sequential
        self.threshold = threshold
        self.timeM = time
        self.fold = folds
        self.repeat = repeats

    def execute_model(self, model_factory_params, X_train, y_train, X_test, y_test, predictions,  customMetric):
        model = ModelFactory().getModel(**model_factory_params)
        preds = []
        accuracy, b_accuracy, f1, custom = 0, 0, 0, 0
        custom = None

        start = time.time()

        model.fit(X=X_train, y=y_train)
        y_pred = model.predict(X=X_test)

        accuracy += accuracy_score(y_test, y_pred, normalize=True)
        b_accuracy += balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        if customMetric is not None:
            custom = customMetric(y_test, y_pred)

        exeT = time.time() - start


        return model_factory_params['Nqubits'], model_factory_params['model'], model_factory_params['Embedding'], model_factory_params['Ansatz'], model_factory_params['Max_features'], exeT, accuracy, b_accuracy, f1, custom, preds
    
    def execute_model_simulation(self, model_factory_params, X_train, y_train, X_test, y_test, predictions,  customMetric):
        #model = ModelFactory().getModel(**model_factory_params)
        preds = []
        accuracy, b_accuracy, f1, custom = 0, 0, 0, 0
        custom = None

        start = time.time()
        print(f"Executing: { model_factory_params['Nqubits']}, {model_factory_params['model']}, {model_factory_params['Embedding']}, {model_factory_params['Ansatz']}, {model_factory_params['Max_features']}")
        exeT = time.time() - start

        return model_factory_params['Nqubits'], model_factory_params['model'], model_factory_params['Embedding'], model_factory_params['Ansatz'], model_factory_params['Max_features'], exeT, accuracy, b_accuracy, f1, custom, preds

    def dispatch(self, nqubits, randomstate, predictions, shots,
                numPredictors, numLayers, classifiers, ansatzs, backend,
                embeddings, features, learningRate, epochs, runs, batch,
                maxSamples, verbose, customMetric, customImputerNum,
                customImputerCat, X_train, y_train, X_test, y_test,
                auto, cores,showTable=True):
        t_pre= time.time()
        combinations = create_combinationsCV(qubits=nqubits,
                                        classifiers=classifiers,
                                        embeddings=embeddings,
                                        features=features,
                                        ansatzs=ansatzs,
                                        RepeatID=[i for i in range(self.repeat)],
                                        FoldID=[i for i in range(self.fold)])



        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)


        # Calculate available memory and determine max parallel models
        available_memory = calculate_free_memory()
        quantum_memory = calculate_quantum_memory(num_qubits=max(nqubits))
        max_models_parallel = min(int(available_memory // quantum_memory) if quantum_memory > 0 else float('inf'), psutil.cpu_count(logical=False))
        print(max_models_parallel)

        # for combination in combinations:
        #     print(combination)

        # Prepare all model executions
        all_executions = []
        cpu_queue = []
        gpu_queue = []
        for combination in combinations:
            qubits, name, embedding, ansatz, feature, repeat, fold = combination
            feature = feature if feature is not None else "~"

            numClasses = len(np.unique(y_train))
            # adjustedQubits = adjustQubits(nqubits=qubits, numClasses=numClasses)
            adjustedQubits = qubits
            # print(f"ADJUSTED QUBITS: {adjustedQubits}")
            prepFactory = PreprocessingFactory(adjustedQubits)
            sanitizer = prepFactory.GetSanitizer(customImputerCat, customImputerNum)

            X_train = pd.DataFrame(sanitizer.fit_transform(X_train))
            X_test = pd.DataFrame(sanitizer.transform(X_test))

            model_factory_params = {
                "Nqubits": adjustedQubits,
                "model": name,
                "Embedding": embedding,
                "Ansatz": ansatz,
                "N_class": numClasses,
                "backend": backend,
                "Shots": shots,
                "seed": randomstate*repeat,
                "Layers": numLayers,
                "Max_samples": maxSamples,
                "Max_features": feature,
                "LearningRate": learningRate,
                "BatchSize": batch,
                "Epoch": epochs,
                "numPredictors": numPredictors
            }
            # print(f"XTEST SHAPE: {X_test.shape}")
            # print(f"YTEST SHAPE: {y_test.shape}")
            # print(f"XTrain SHAPE: {X_train.shape}")
            # print(f"YTrain SHAPE: {y_train.shape}")
            preprocessing = prepFactory.GetPreprocessing(ansatz=ansatz, embedding=embedding)
            X_train_processed = np.array(preprocessing.fit_transform(X_train, y=y_train))
            X_test_processed = np.array(preprocessing.transform(X_test))
            y_test = np.array(y_test)
            y_train = np.array(y_train)
            # print(f"XTEST PROCESSED SHAPE: {X_test_processed.shape}")
            # print(f"XTrain PROCESSED SHAPE: {X_train_processed.shape}")

            
            #all_executions.append((model_factory_params, X_train_processed, y_train, X_test_processed, y_test, predictions, customMetric))
            
            if name == Model.QNN and qubits>=self.threshold:
                gpu_queue.append((model_factory_params, X_train_processed, y_train, X_test_processed, y_test, predictions, customMetric))
            else:
                cpu_queue.append((model_factory_params, X_train_processed, y_train, X_test_processed, y_test, predictions, customMetric))
        
        if self.timeM:
            print(f"PREPROCESSING TIME: {time.time()-t_pre}")
        # Execute all models in parallel
        t_exe = time.time()
        if self.sequential:
            print(f"CPU QUEUE:")
            results_cpu = [self.execute_model_simulation(*execution_params) for execution_params in cpu_queue]
            
            print(f"GPU QUEUE:")
            results_gpu = [self.execute_model_simulation(*execution_params) for execution_params in gpu_queue]
            
        else:
            if auto:
                results = Parallel(n_jobs=cores, prefer='processes',batch_size='auto',verbose=10)(
                    delayed(self.execute_model)(*execution_params) for execution_params in all_executions
                )
            else:
                results = Parallel(n_jobs=cores, prefer='processes',batch_size=1,verbose=10)(
                    delayed(self.execute_model)(*execution_params) for execution_params in all_executions
                )
        if self.timeM:
            print(f"EXECUTING TIME: {time.time()-t_exe}")
        # Process results
        t_res = time.time()
        scores = pd.DataFrame(results, columns=["Qubits","Model", "Embedding", "Ansatz", "Features", "Time taken", "Accuracy", "Balanced Accuracy", "F1 Score", "Custom Metric", "Predictions"])
        # if showTable:
        #     print(scores.to_markdown())

        # if self.timeM:
        #     print(f"RESULTS TIME: {time.time()-t_res}")
        # return scores
