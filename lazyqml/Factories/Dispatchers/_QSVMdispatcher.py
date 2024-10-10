from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, roc_auc_score
from Factories.Models.fModels import *
from Factories.Preprocessing.fPreprocessing import *
import time 
import pandas as pd

def executeQSVM(combinations, nqubits, backend, shots, runs, X_train, y_train, X_test, y_test, customImputerCat,customImputerNum, customMetric,seed, predictions,numClasses):
    
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
    
    
    # Convert input data to pandas DataFrames if they aren't already
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test)
    
    
    prepFactory = PreprocessingFactory(nqubits)
    sanitizer = prepFactory.GetSanitizer(customImputerCat, customImputerNum)
    
    X_train = sanitizer.fit_transform(X_train)
    X_test = sanitizer.transform(X_test)

    for i in combinations:
        print(f"Executing {i} combination of {len(combinations)}")
        name, embedding, ansatz, feature = i
        model = ModelFactory(Nqubits=nqubits, Embedding=embedding,Ansatz=ansatz,backend=backend,Shots=shots,seed=seed,N_class=numClasses).GetQSVM()
        
        preprocessing = None
        preprocessing = prepFactory.GetPreprocessing(embedding=embedding,ansatz=ansatz)

        X_train = preprocessing.fit_transform(X=X_train,y=y_train)
        X_test = preprocessing.transform(X=X_test)

        accuracyR, b_accuracyR, f1R = 0, 0, 0
        for j in range(runs):
            print(f"Executing {j+1} run of {runs}")
            start = time.time()
            model.fit(X=X_train,y=y_train)
            exeT = time.time() - start
            y_pred = model.predict(X=X_test)

            accuracyR += accuracy_score(y_test, y_pred, normalize=True)
            b_accuracyR += balanced_accuracy_score(y_test, y_pred)
            f1R += f1_score(y_test, y_pred, average="weighted")
            # try:
            #         roc_aucR += roc_auc_score(y_test, y_pred)
            # except Exception as exception:
            #         roc_aucR += ""
        
        accuracy = accuracyR/runs
        b_accuracy = b_accuracyR/runs
        f1 = f1R/runs
        # roc_auc = roc_aucR/runs

        NAMES.append(name)
        ANSATZ.append(ansatz)
        ACCURACY.append(accuracy)
        B_ACCURACY.append(b_accuracy)
        # ROC_AUC.append(roc_auc)
        EMBEDDINGS.append(embedding)
        F1.append(f1)
        TIME.append(exeT)
        PARAMETERS.append("~")
        
        if customMetric is not None:
            customMetricV = customMetric(y_test, y_pred)
            customMetric.append(customMetricV)
            scores = pd.DataFrame(
                {
                    "Model": NAMES,
                    "Embedding": EMBEDDINGS,
                    "Ansatz": ANSATZ,
                    "Accuracy": ACCURACY,
                    "Balanced Accuracy": B_ACCURACY,
                    # "ROC AUC": ROC_AUC,
                    "F1 Score": F1,
                    customMetric.__name__: customMetric,
                    "Trainable Parameters": PARAMETERS,
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
                        # "ROC AUC": ROC_AUC,
                        "F1 Score": F1,
                        "Trainable Parameters": PARAMETERS,
                        "Time taken": TIME,
                    }
                )
            
    return scores

