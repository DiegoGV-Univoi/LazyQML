from Utils.Utils import * 
import numpy as np
import pandas as pd
import math
from Factories.Models.fModels import *
from Factories.Preprocessing.fPreprocessing import *
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
import time

class Dispatcher:
    def __init__(self, sequential = False, threshold=27):
        self.sequential = sequential,
        self.threshold = threshold

    def executeModel(self, model,X_train,y_train,X_test,y_test,predictions,runs,customMetric):
        preds = []
        accuracyR, b_accuracyR, f1R, customR = 0, 0, 0, 0
        custom = None
        for j in range(runs):
            printer.print(f"Executing {j+1} run of {runs}")
            start = time.time()
            model.fit(X=X_train,y=y_train)
            exeT = time.time() - start
            y_pred = model.predict(X=X_test)

            accuracyR += accuracy_score(y_test, y_pred, normalize=True)
            b_accuracyR += balanced_accuracy_score(y_test, y_pred)
            f1R += f1_score(y_test, y_pred, average="weighted")
            if customMetric is not None:
                customR += customMetric(y_test,y_pred)
            # try:
            #         roc_aucR += roc_auc_score(y_test, y_pred)
            # except Exception as exception:
            #         roc_aucR += ""
        if predictions:
            preds = y_pred
        accuracy = accuracyR/runs
        b_accuracy = b_accuracyR/runs
        f1 = f1R/runs
        if customMetric is not None:
            custom = customR/runs
        # roc_auc = roc_aucR/runs

        return exeT, accuracy, b_accuracy, f1, custom, preds


    def dispatch(self, nqubits, randomstate, predictions,  shots, #ignoreWarnings,
                    numPredictors, numLayers, classifiers, ansatzs, backend, 
                    embeddings, features, learningRate, epochs, runs, batch,
                    maxSamples, verbose, customMetric, customImputerNum, customImputerCat, X_train, y_train, X_test, y_test,showTable=True,sequential=True,threshold=26):
        
        NAMES = []
        EMBEDDINGS = []
        ANSATZ = []
        ACCURACY = []
        B_ACCURACY = []
        FEATURES  = []
        #ROC_AUC = []
        F1 = []
        TIME = []
        CUSTOM = []
        #PARAMETERS = []
        

        numClasses = len(np.unique(y_train))

        combinations = create_combinations(classifiers=classifiers,embeddings=embeddings,features=features,ansatzs=ansatzs)
        
        if (numClasses > 2**math.floor(math.log2(nqubits))):
            printer.print("The number of qubits must exceed the number of classes and be a power of 2 to execute all circuits successfully. \nEnsure that nqubits > #classes and that 2^floor(log2(nqubits)) > #classes.\nThe number of qubits will be changed to a valid one, this change will affect the QuantumClassifier object.")
            adjustedQubits = adjustQubits(nqubits=nqubits,numClasses=numClasses)
            printer.print(f"New number of qubits:\t{adjustedQubits}")
        else:
            adjustedQubits = nqubits
    
        scores = pd.DataFrame()

        # Convert input data to pandas DataFrames if they aren't already
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train)
        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test)

        prepFactory = PreprocessingFactory(nqubits)
        sanitizer = prepFactory.GetSanitizer(customImputerCat, customImputerNum)

        X_train = sanitizer.fit_transform(X_train)
        X_test = sanitizer.transform(X_test)

        print(combinations)
        for combination in combinations:
            name, embedding, ansatz, feature = combination
            printer.print(f"Model: {name} Embedding: {embedding} Ansatz:{ansatz}")
            model = ModelFactory().getModel(Nqubits=adjustedQubits, model=name, Embedding=embedding,Ansatz=ansatz, N_class=numClasses,backend=backend,Shots=shots,seed=randomstate,Layers=numLayers,Max_samples=maxSamples,Max_features=feature,LearningRate=learningRate,BatchSize=batch,Epoch=epochs,numPredictors=numPredictors)
            preprocessing = prepFactory.GetPreprocessing(ansatz=ansatz,embedding=embedding)
            X_train=preprocessing.fit_transform(X_train,y=y_train)
            X_test=preprocessing.transform(X_test)
            exeT, accuracy, b_accuracy, f1, custom, preds = self.executeModel(model, X_train, y_train, X_test, y_test,predictions,runs=runs,customMetric=customMetric)
            
            NAMES.append(name)
            ANSATZ.append(ansatz)
            ACCURACY.append(accuracy)
            B_ACCURACY.append(b_accuracy)
            #ROC_AUC.append(roc_auc)
            EMBEDDINGS.append(embedding)
            F1.append(f1)
            TIME.append(exeT)
            FEATURES.append(feature)
            CUSTOM.append(custom)
            #PARAMETERS.append(trainable)


        if customMetric is None:
            scores = pd.DataFrame(
                {
                    "Model": NAMES,
                    "Embedding": EMBEDDINGS,
                    "Ansatz": ANSATZ,
                    "Features": FEATURES,
                    "Accuracy": ACCURACY,
                    "Balanced Accuracy": B_ACCURACY,
                    #"ROC AUC": ROC_AUC,
                    "F1 Score": F1,
                    #"Trainable Parameters": PARAMETERS,
                    "Time taken": TIME,
                }
            )
        else:
            scores = pd.DataFrame(
                {
                    "Model": NAMES,
                    "Embedding": EMBEDDINGS,
                    "Ansatz": ANSATZ,
                    "Features": FEATURES,
                    "Accuracy": ACCURACY,
                    "Balanced Accuracy": B_ACCURACY,
                    #"ROC AUC": ROC_AUC,
                    "F1 Score": F1,
                    customMetric.__name__: CUSTOM,
                    #"Trainable Parameters": PARAMETERS,
                    "Time taken": TIME,
                }
            )
        if showTable:
            print(scores.to_markdown())