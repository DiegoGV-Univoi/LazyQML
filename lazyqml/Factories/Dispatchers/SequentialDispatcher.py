from Utils.Utils import * 
import numpy as np
from Factories.Dispatchers.QSVMdispatcher import *
from Factories.Dispatchers.QNNdispatcher import *
from Factories.Dispatchers.QNNBagdispatcher import *
import math

def dispatch(nqubits, randomstate, predictions,  shots, #ignoreWarnings,
                 numPredictors, numLayers, classifiers, ansatzs, backend,
                 embeddings, features, learningRate, epochs, runs,
                 maxSamples, verbose, customMetric, customImputerNum, customImputerCat, X_train, y_train, X_test, y_test):
    
    numClasses = len(np.unique(y_train))

    fixSeed(seed=randomstate)

    QSVM, QNN, QNNBAG = create_enum_combinations(classifiers=classifiers,embeddings=embeddings,features=features,ansatzs=ansatzs)
    
    if (numClasses > 2**math.floor(math.log2(nqubits))):
        print("The number of qubits must exceed the number of classes and be a power of 2 to execute all circuits successfully. \nEnsure that nqubits > #classes and that 2^floor(log2(nqubits)) > #classes.\nThe number of qubits will be changed to a valid one, this change will affect the QuantumClassifier object.")
        adjustedQubits = adjustQubits(nqubits=nqubits,numClasses=numClasses)
        print(f"New number of qubits:\t{adjustedQubits}")
    else:
        adjustedQubits = nqubits
   
    scores = pd.DataFrame()


    if QSVM:
        scores = executeQSVM(combinations=QSVM,nqubits=adjustedQubits,backend=backend,runs=runs,shots=shots,X_train=X_train,y_train=y_train,X_test=X_test, y_test=y_test,customImputerCat=customImputerCat,customImputerNum=customImputerNum,seed=randomstate, predictions=predictions,customMetric=customMetric,numClasses=numClasses)
    if QNN:
        executeQNN()
    if QNNBAG:
        executeQSVM()
    
    print(scores.to_markdown())

    pass
