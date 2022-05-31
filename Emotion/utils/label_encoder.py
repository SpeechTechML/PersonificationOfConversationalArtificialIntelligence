import os
import pickle

from sklearn.preprocessing import LabelEncoder


class LabelEncoderWithFitCheck(LabelEncoder):
    def __init__(self, save_dir: str, *args, **kwargs):
        super(LabelEncoderWithFitCheck, self).__init__(*args, **kwargs)
        self._save_dir = save_dir
        self._is_fitted = False
        
    @property
    def is_fitted(self):
        return self._is_fitted
    
    def fit(self, y):
        self._is_fitted = True
        return super().fit(y)

    def fit_transform(self, y):
        return super().fit_transform(y)

    def save(self):
        save_path = os.path.join(self._save_dir, 'label_encoder.pkl')
        with open(save_path, 'wb') as file:
            pickle.dump(self, file)
