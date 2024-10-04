import inspect
import logging
import warnings
from tabulate import tabulate

def check_classifiers(self,array):
    allowed_keywords = {"all", "qsvm", "qnn", "qnn_bag"} 
    return all(keyword in allowed_keywords for keyword in array)
def check_embeddings(self,array):
    allowed_keywords = {"all", "amplitude_embedding", "ZZ_embedding", "rx_embedding", "rz_embedding", "ry_embedding"} 
    return all(keyword in allowed_keywords for keyword in array)
def check_ansatz(self,array):
    allowed_keywords = {"all", "HPzRx","hardware_efficient", "tree_tensor", "two_local"} 
    return all(keyword in allowed_keywords for keyword in array)
def check_features(self,array):
    return all((keyword > 0 and keyword <= 1) for keyword in array)

class Validator:
    def __init__(self) -> None:
        pass


    def validate(self, nqubits, randomstate, predictions, ignoreWarnings, numPredictors, numLayers, customMetric, customImputerNum, customImputerCat, classifiers, ansatzs, embeddings,features,verbose,learningRate,epochs,runs,maxSamples):
        errors = 0
        errormsg = []
        
        if isinstance(nqubits, int):
            if nqubits <= 0:
                errors += 1
                errormsg.append("The parameter <nqubits> should have a value greater than 0.")
            else:
                self.nqubits = nqubits
        else:
            errors += 1
            errormsg.append("The parameter <nqubits> should be of integer type.")
        
        if isinstance(randomstate, int):        
            self.randomstate = randomstate
        else:
            errors += 1
            errormsg.append("The parameter <randomstate> should be of integer type.")

        if isinstance(predictions, bool):        
            self.predictions = predictions
        else:
            errors += 1
            errormsg.append("The parameter <predictions> should be of boolean type.")
        
        if isinstance(ignoreWarnings, bool):        
            self.ignoreWarnings = ignoreWarnings
        else:
            errors += 1
            errormsg.append("The parameter <ignoreWarnings> should be of boolean type.")

        if isinstance(numLayers, int):        
            self.numLayers = numLayers
        else:
            errors += 1
            errormsg.append("The parameter <numLayers> should be of integer type.")

        if isinstance(numPredictors, int):
            if numPredictors > 0:        
                self.numPredictors = numPredictors
            else:
                errors += 1
                errormsg.append("The parameter <numPredictors> should be bigger than 0.")    
        else:
            errors += 1
            errormsg.append("The parameter <numPredictors> should be of integer type.")

        are_all_strings = all(isinstance(element, str) for element in classifiers)
        if are_all_strings:
            if check_classifiers(classifiers):
                self.classifiers = classifiers
            else:
                errors += 1
                errormsg.append("The parameter <classifiers> should belong to the following list ['qsvm', 'qnn', 'qnn_bag'].")
        else:
            errors += 1
            errormsg.append("The parameter <classifiers> should be a list of strings type.")        

        are_all_strings = all(isinstance(element, str) for element in embeddings)
        if are_all_strings:
            if check_embeddings(embeddings):
                self.embeddings = embeddings
            else:
                errors += 1
                errormsg.append("The parameter <embeddings> should belong to the following list ['amplitude_embedding', 'ZZ_embedding', 'rx_embedding', 'rz_embedding', 'ry_embedding']")
        else:
            errors += 1
            errormsg.append("The parameter <embeddings> should be a list of strings type.")            
        
        are_all_strings = all(isinstance(element, str) for element in ansatzs)
        if are_all_strings:
            if check_ansatz(ansatzs):
                self.ansatzs = ansatzs
            else:
                errors += 1
                errormsg.append("The parameter <ansatzs> should belong to the following list ['HPzRx', 'tree_tensor'', 'two_local', 'hardware_efficient']")
        else:
            errors += 1
            errormsg.append("The parameter <ansatzs> should be a list of strings type.")  

        are_all_strings = all(isinstance(element, float) for element in features)
        if are_all_strings:
            if check_features(features):
                self.features = features
            else:
                errors += 1
                errormsg.append("The parameter <features> should belong to the interval (0,1].")
        else:
            errors += 1
            errormsg.append("The parameter <features> should be a list of floats type.")     

        if isinstance(learningRate, float):
            if learningRate > 0:
                self.learningRate = learningRate
            else:
                errors += 1
                errormsg.append("The parameter <learningRate> should be bigger than 0.")
        else:
            errors += 1
            errormsg.append("The parameter <learningRate> should be of float type.")        

        if isinstance(epochs, int):
            if epochs > 0:
                self.epochs = epochs
            else:
                errors += 1
                errormsg.append("The parameter <epochs> should be bigger than 0.")
        else:
            errors += 1
            errormsg.append("The parameter <epochs> should be of integer type.")      

        if isinstance(runs,int):
            if runs > 0:
                self.runs = runs
            else:
                errors += 1
                errormsg.append("The parameter <runs> should be bigger than 0.")
        else:
            errors += 1
            errormsg.append("The parameter <runs> should be of integer type.")      

        self.maxSamples = maxSamples

        if isinstance(verbose,bool):
            self.verbose = verbose
            self.verboseprint = print if verbose else lambda *a, **k: None
        else:
            logging.warn("Verbose is not an instance of bool, False will be assumed.")
            self.verboseprint = lambda *a, **k: None
        
        self.optimizer = optax.adam(learning_rate=self.learningRate)
        
        if customImputerNum is not None:
            module = inspect.getmodule(customImputerNum)
            # Check if the module belongs to sklearn.impute
            if module.__name__.startswith('sklearn.impute'):
                logging.warn("The object belongs to the sklearn.impute module. Custom Numeric Imputer will be used.")
                self.numeric_transformer = Pipeline(
                steps=[("imputer",customImputerNum), ("scaler", StandardScaler())])
            else:
                logging.warn("The object does not belong to the sklearn.impute module. Default Custom Numeric Imputer will be used.")
                self.numeric_transformer = Pipeline(
                steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())])
        else:
            self.numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())])


        if customImputerCat is not None:
            module = inspect.getmodule(customImputerCat)
            # Check if the module belongs to sklearn.impute
            if module.__name__.startswith('sklearn.impute'):
                logging.warn("The object belongs to the sklearn.impute module. Custom Categorical Imputer will be used.")
                self.categorical_transformer = Pipeline(
                steps=[("imputer",customImputerCat), ("scaler", StandardScaler())])
            else:
                logging.warn("The object does not belong to the sklearn.impute module. Default Custom Categorical Imputer will be used.")
                self.categorical_transformer = Pipeline(
                steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())])
        else:    
            self.categorical_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())])
        
        if is_metric_function(customMetric):
            self.customMetric = customMetric
        else:
            self.customMetric = None

        if errors > 0:
            for i in errormsg:
                logging.error(i,exc_info=False)
            exit()
        print(tabulate([[self.learningRate,self.epochs,self.runs,self.classifiers,self.embeddings,self.ansatzs,self.features,self.nqubits,self.numLayers,self.numPredictors]], headers=['Learning Rate', 'Epochs', "# Runs","Classifiers","Embeddings","Ansatzs","Features","Qubits","Layers","Predictors"], tablefmt='orgtbl'))