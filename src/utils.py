from datetime import datetime
import torch
from sklearn.metrics import cohen_kappa_score
import scipy as sp
import numpy as np
from functools import partial


def getDevice():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_model_stamp():
    return datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

def get_actual_predictions(preds, coeff=[0.5, 1.5, 2.5, 3.5]):
    device = getDevice()
    actual_preds = torch.zeros(preds.shape, device=device)
    for i, p in enumerate(preds):
        if p < coeff[0]:
            ap = 0
        elif p < coeff[1]:
            ap = 1
        elif p < coeff[2]:
            ap = 2
        elif p < coeff[3]:
            ap = 3
        else:
            ap = 4
        actual_preds[i] = torch.tensor(ap, device=device, dtype=torch.float)
    return actual_preds

class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4

        ll = cohen_kappa_score(y, X_p, weights='quadratic')
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p

    def coefficients(self):
        return self.coef_['x']

if __name__ == "__main__":
    scores = [0.3, 1.67, 2.3, 0.6, 0.75, 2.45, 3.2, 3.3, 3.7, 8, 0.3, 1.67, 2.3, 0.6, 0.75, 2.45, 3.2, 3.3, 3.7, 8, 0.3, 1.67, 2.3, 0.6, 0.75, 2.45, 3.2, 3.3, 3.7, 8]
    target = [0, 2, 2, 0, 1, 3, 2, 4, 4, 4, 0, 2, 2, 0, 1, 3, 2, 4, 4, 4, 0, 2, 2, 0, 1, 3, 2, 4, 4, 4]
    optR = OptimizedRounder()
    optR.fit(scores,target)

    coeff = optR.coefficients()
    print("Learned coeff: ", coeff)

    preds = optR.predict(scores, coeff)
    correct = sum(np.array(preds) == np.array(target))
    print(correct, len(scores))
