from pytensorforge.backend import xp
class StandardScaler:
    def fit(self, X):
        self.mean = xp.mean(X, axis=0)
        self.std = xp.std(X, axis=0)
        self.std[self.std == 0] = 1.0

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)