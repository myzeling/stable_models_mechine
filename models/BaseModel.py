class BaseModel:
    def __init__(self,args):
        self.model = None
        self.args = args
    def fit(self):
        raise NotImplementedError

    def predict(self, data):
        return self.model.predict(data)

def H(x, xi):
    if abs(x) <= xi:
        return x**2
    else:
        return 2*xi*abs(x) - xi**2