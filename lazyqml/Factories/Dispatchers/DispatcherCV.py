from Utils.Utils import *
import numpy as np
import pandas as pd
import math
from Factories.Models.fModels import *
from Factories.Preprocessing.fPreprocessing import *
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
import time
from joblib import Parallel, delayed
from multiprocessing import Queue, Process, Pool, Manager
import multiprocessing as mp
import os
from Utils.Utils import printer
from statistics import mean
from collections import defaultdict
from sklearn.model_selection import RepeatedStratifiedKFold, LeaveOneOut
import torch


def generate_cv_indices(X, y, mode="cross-validation", test_size=0.4, n_splits=5, n_repeats=1, random_state=None):
    """
    Generate train and test indices for either cross-validation, holdout split, or leave-one-out.
    
    Parameters:
        X (pd.DataFrame or np.ndarray): The features matrix.
        y (pd.Series or np.ndarray): The target vector.
        mode (str): "cross-validation", "holdout", or "leave-one-out".
        test_size (float): Test set proportion for the holdout split (ignored for CV and LOO).
        n_splits (int): Number of folds in StratifiedKFold (ignored for holdout and LOO).
        n_repeats (int): Number of repeats for cross-validation (ignored for holdout and LOO).
        random_state (int): Random state for reproducibility.
    
    Returns:
        dict: A dictionary of train/test indices.
    """
    cv_indices = {}
    
    if mode == "holdout":
        # Single train-test split for holdout
        train_idx, test_idx = train_test_split(
            np.arange(len(X)),
            test_size=test_size,
            stratify=y,
            random_state=random_state
        )
        cv_indices[(0, 0)] = {
            'train_idx': train_idx,
            'test_idx': test_idx
        }
    
    elif mode == "cross-validation":
        # StratifiedKFold for cross-validation splits
        for repeat in range(n_repeats):
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state + repeat if random_state is not None else None)
            for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                cv_indices[(repeat, fold)] = {
                    'train_idx': train_idx,
                    'test_idx': test_idx
                }
    
    elif mode == "leave-one-out":
        # LeaveOneOut cross-validation
        loo = LeaveOneOut()
        for fold, (train_idx, test_idx) in enumerate(loo.split(X)):
            cv_indices[(0, fold)] = {
                'train_idx': train_idx,
                'test_idx': test_idx
            }
    
    else:
        raise ValueError("Invalid mode. Choose 'holdout', 'cross-validation', or 'leave-one-out'.")
    
    return cv_indices

def get_train_test_split(cv_indices, repeat_id=0, fold_id=0):
    """
    Retrieve the train and test indices for a given repeat and fold ID.
    
    Parameters:
        cv_indices (dict): The cross-validation indices dictionary.
        repeat_id (int): The repeat ID (0 to n_repeats-1 or 0 for holdout/LOO).
        fold_id (int): The fold ID within the specified repeat.
    
    Returns:
        tuple: (train_idx, test_idx) arrays for the specified fold and repeat.
    """
    indices = cv_indices.get((repeat_id, fold_id))
    if indices is None:
        print(f"RepeatID {repeat_id}, FoldID{fold_id}")
        raise ValueError("Invalid repeat_id or fold_id specified.")
    
    return indices['train_idx'], indices['test_idx']



def dataProcessing(X, y, prepFactory, customImputerCat, customImputerNum, 
                  train_idx, test_idx, ansatz=None, embedding=None):
    """
    Process data for specific train/test indices.
    
    Parameters:
    - X: Input features
    - y: Target variable
    - prepFactory: Preprocessing factory object
    - customImputerCat: Categorical imputer
    - customImputerNum: Numerical imputer
    - train_idx: Training set indices
    - test_idx: Test set indices
    - ansatz: Optional preprocessing ansatz
    - embedding: Optional embedding method
    
    Returns:
    Tuple of (X_train_processed, X_test_processed, y_train, y_test)
    """
    # Split the data using provided indices
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Create sanitizer and preprocess
    sanitizer = prepFactory.GetSanitizer(customImputerCat, customImputerNum)
    X_train = pd.DataFrame(sanitizer.fit_transform(X_train))
    X_test = pd.DataFrame(sanitizer.transform(X_test))
    
    # Apply additional preprocessing if ansatz/embedding provided
    if ansatz is not None or embedding is not None:
        preprocessing = prepFactory.GetPreprocessing(ansatz=ansatz, embedding=embedding)
        X_train_processed = np.array(preprocessing.fit_transform(X_train, y=y_train))
        X_test_processed = np.array(preprocessing.transform(X_test))
    else:
        X_train_processed = np.array(X_train)
        X_test_processed = np.array(X_test)
    
    # Convert target variables to numpy arrays
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    return X_train_processed, X_test_processed, y_train, y_test



class DispatcherCV:
    def __init__(self, sequential=False, threshold=22, time=True, folds=10, repeats=5):
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

    
    def process_gpu_task(self, queue, results):
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        
        while not queue.empty():
            try:
                item = queue.get_nowait()
                
                results.append(self.execute_model(*item[1]))  # Store results if needed
            except queue.Empty:
                break


    def process_cpu_task(self,cpu_queue, gpu_queue, results):
        torch.set_num_interop_threads(1)
        torch.set_num_threads(1)
        numProcs = psutil.cpu_count(logical=False)
        total_memory = calculate_free_memory()
        available_memory = total_memory
        available_cores = numProcs
        
        # Lock para el acceso seguro a los recursos compartidos
        manager = Manager()
        resource_lock = manager.Lock()

        while not cpu_queue.empty():
            try:
                # Determinar número de cores a usar basado en el estado de gpu_queue
                if gpu_queue.empty():
                    max_cores = numProcs
                else:
                    max_cores = max(1, numProcs - 1)
                
                current_batch = []
                current_cores = 0
                
                # Recolectar items para procesar mientras haya recursos disponibles
                while current_cores < max_cores and not cpu_queue.empty():
                    try:
                        item = cpu_queue.get_nowait()
                        # printer.print(f"ITEM CPU: {item[0]}")
                        _, _, _, _, _, _, _, mem_model = item[0]
                        
                        # Verificar si hay recursos suficientes
                        with resource_lock:
                            if available_memory >= mem_model and available_cores >= 1:
                                # printer.print(f"Available Resources - Memory: {available_memory}, Cores: {available_cores}")
                                available_memory -= mem_model
                                available_cores -= 1
                                current_batch.append(item)
                                current_cores += 1
                            else:
                                # printer.print(f"Unavailable Resources - Requirements: {mem_model}, Available: {available_memory}")
                                cpu_queue.put(item)
                                break
                    
                    except Queue.Empty:
                        break
                
                # Procesar el batch actual si no está vacío
                if current_batch:
                    # printer.print(f"Executing Batch of {len(current_batch)} Jobs")
                    with Pool(processes=len(current_batch)) as pool:
                        # Usamos map de forma síncrona para asegurar que todos los items se procesen
                        batch_results = pool.starmap(self.execute_model, [params[1] for params in current_batch])
                        
                        # Filtramos los resultados None (errores) y los añadimos a results
                        # valid_results = [r for r in batch_results if r is not None]
                        results.extend(batch_results)
                    
                    # Liberar recursos después del procesamiento
                    with resource_lock:
                        # printer.print("Freeing Up Resources")
                        for item in current_batch:
                            _, _, _, _, _, _, _, mem_model = item[0]
                            available_memory += mem_model
                            available_cores += 1
                            # printer.print(f"Freed - Memory: {available_memory}MB, Cores: {available_cores}")
                
                # printer.print("Waiting for next batch")
                time.sleep(0.1)
            
            except Exception as e:
                printer.print(f"Error in the batch: {str(e)}")
                break

    def dispatch(self, nqubits, randomstate, predictions, shots,
                numPredictors, numLayers, classifiers, ansatzs, backend,
                embeddings, features, learningRate, epochs, runs, batch,
                maxSamples, verbose, customMetric, customImputerNum,
                customImputerCat, X, y, 
                showTable=True, mode="cross-validation",testsize=0.4):

        """
        ################################################################################
        Preparing Data Structures & Initializing Variables
        ################################################################################
        """
        # Replace the list-based queues with multiprocessing queues
        manager = Manager()
        gpu_queue = Queue()
        cpu_queue = Queue()
        results = manager.list()  # Shared list for results if needed      
        # Also keep track of items for printing
        cpu_items = []
        gpu_items = []

        RAM = calculate_free_memory()
        VRAM = calculate_free_video_memory()
         
        """
        ################################################################################
        Generate CV indices once
        ################################################################################
        """
        cv_indices = generate_cv_indices(
            X, y, 
            mode=mode, 
            n_splits=self.fold, 
            n_repeats=self.repeat, 
            random_state=randomstate
        )
    
        
        """
        ################################################################################
        Generating Combinations
        ################################################################################
        """

        t_pre = time.time()
        combinations = create_combinations(qubits=nqubits,
                                        classifiers=classifiers,
                                        embeddings=embeddings,
                                        features=features,
                                        ansatzs=ansatzs,
                                        RepeatID=[i for i in range(self.repeat)],
                                        FoldID=[i for i in range(self.fold)])
        cancelledQubits = set()
        to_remove = []
    
        for i, combination in enumerate(combinations):
            modelMem = combination[-1]
            if modelMem > RAM and modelMem > VRAM:
                to_remove.append(combination)

        for combination in to_remove:
            combinations.remove(combination)
            cancelledQubits.add(combination[0])

        for val in cancelledQubits:
            printer.print(f"Execution with {val} Qubits are cancelled due to memory constrains -> Memory Required: {calculate_quantum_memory(val)/1024:.2f}GB Out of {calculate_free_memory()/1024:.2f}GB")

        X = pd.DataFrame(X)

        # Prepare all model executions
        for combination in combinations:
            qubits, name, embedding, ansatz, feature, repeat, fold, memModel = combination
            feature = feature if feature is not None else "~"

            # Get indices for this repeat/fold combination
            train_idx, test_idx = get_train_test_split(cv_indices, repeat, fold)
            
            numClasses = len(np.unique(y))
            adjustedQubits = qubits  # or use adjustQubits if needed
            prepFactory = PreprocessingFactory(adjustedQubits)
            
            # Process data for this specific combination using pre-generated indices
            X_train_processed, X_test_processed, y_train_processed, y_test_processed = dataProcessing(
                X, 
                y, 
                prepFactory, 
                customImputerCat, 
                customImputerNum,
                train_idx,
                test_idx,
                ansatz=ansatz,
                embedding=embedding
            )

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

            # When adding items to queues
            if name == Model.QNN and qubits >= self.threshold and VRAM > calculate_quantum_memory(qubits):
                model_factory_params["backend"] = Backend.lightningGPU
                gpu_queue.put((combination,(model_factory_params, X_train_processed, y_train_processed, X_test_processed, y_test_processed, predictions, customMetric)))
                gpu_items.append(combination)
            else:
                model_factory_params["backend"] = Backend.lightningQubit
                cpu_queue.put((combination,(model_factory_params, X_train_processed, y_train_processed, X_test_processed, y_test_processed, predictions, customMetric)))
                cpu_items.append(combination)

        if self.timeM:
            printer.print(f"PREPROCESSING TIME: {time.time()-t_pre}")
        
        """
        ################################################################################
        Creating processes
        ################################################################################
        """
        executionTime = time.time()
        gpu_process = None
        # Start GPU process
        if not gpu_queue.empty():
            gpu_process = Process(target=self.process_gpu_task, args=(gpu_queue, results))
            gpu_process.start()

        # Start CPU processes
        if not cpu_queue.empty():
            self.process_cpu_task(cpu_queue, gpu_queue, results)
        
        # Wait for all processes to complete
        if gpu_process is not None:
            gpu_process.join()
        
        executionTime = time.time()-executionTime
        printer.print(f"Execution TIME: {executionTime}")

        """
        ################################################################################
        Processing results
        ################################################################################
        """
        t_res = time.time()

        grouped_results = defaultdict(list)
        for result in list(results):
            key = result[:5]
            grouped_results[key].append(result)

        summary = []
        for key, group in grouped_results.items():
            time_taken = sum([r[5] for r in group])
            accuracy = mean([r[6] for r in group])
            balanced_accuracy = mean([r[7] for r in group])
            f1_score = mean([r[8] for r in group])
            if customMetric:
                custom_metric = mean([r[9] for r in group])
                summary.append((key[0], key[1], key[2], key[3], key[4], time_taken, accuracy, balanced_accuracy, f1_score, custom_metric, []))
            else:
                summary.append((key[0], key[1], key[2], key[3], key[4], time_taken, accuracy, balanced_accuracy, f1_score, [], []))

        scores = pd.DataFrame(summary, columns=["Qubits", "Model", "Embedding", "Ansatz", "Features", "Time taken", "Accuracy", "Balanced Accuracy", "F1 Score", "Custom Metric", "Predictions"])
        if showTable:
            print(scores.to_markdown())

        if self.timeM:
            printer.print(f"RESULTS TIME: {time.time() - t_res}")
        return scores