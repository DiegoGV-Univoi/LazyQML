from Interfaces.iModel import Model

class QNN(Model):

    def fit(X, y, self):
        return super().fit(y, self)
    
    def predict(X, self):
        return super().predict(self)