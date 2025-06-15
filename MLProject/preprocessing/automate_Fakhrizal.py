from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, PolynomialFeatures

class SklearnPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, num_columns, ordinal_columns, nominal_columns, degree=2):
        self.num_columns = num_columns
        self.ordinal_columns = ordinal_columns
        self.nominal_columns = nominal_columns
        self.degree = degree

    def fit(self, X, y=None):
        self.num_pipeline = Pipeline([
            ('poly', PolynomialFeatures(degree=self.degree, include_bias=False)),
            ('scaler', StandardScaler())
        ])

        self.ordinal_pipeline = Pipeline([
            ('ord_encoder', OrdinalEncoder())
        ])

        self.nominal_pipeline = Pipeline([
            ('nom_encoder', OneHotEncoder(drop='first', sparse_output=False))
        ])

        self.preprocessor = ColumnTransformer([
            ('num_pipeline', self.num_pipeline, self.num_columns),
            ('ordinal_pipeline', self.ordinal_pipeline, self.ordinal_columns),
            ('nominal_pipeline', self.nominal_pipeline, self.nominal_columns)
        ]).set_output(transform='pandas')

        self.preprocessor.fit(X)
        return self

    def transform(self, X):
        return self.preprocessor.transform(X)