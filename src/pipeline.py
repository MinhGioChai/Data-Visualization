import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from pathlib import Path
import joblib


class BasicImputerTransformer(BaseEstimator, TransformerMixin):
    """Handle basic missing value imputation for specific columns"""
    
    def __init__(self):
        self.annuity_median_ = None
        self.fam_members_median_ = None
        self.phone_change_median_ = None
        self.org_type_mode_ = None
        self.gender_mode_ = None
    
    def fit(self, X, y=None):
        # Fit statistics that will be used in transform
        self.annuity_median_ = X['AMT_ANNUITY'].median()
        self.fam_members_median_ = X['CNT_FAM_MEMBERS'].median()
        self.phone_change_median_ = X['DAYS_LAST_PHONE_CHANGE'].median()
        self.org_type_mode_ = X['ORGANIZATION_TYPE'].mode()[0] if 'ORGANIZATION_TYPE' in X.columns else None
        self.gender_mode_ = X['CODE_GENDER'].mode()[0] if 'CODE_GENDER' in X.columns else None
        return self
    
    def transform(self, X):
        X = X.copy()
        # CODE_GENDER
        #update 2811: impute code gender with mode instead of dropping rows
        X.loc[X['CODE_GENDER'] == 'XNA', 'CODE_GENDER'] = self.gender_mode_
        # NAME_TYPE_SUITE
        X['NAME_TYPE_SUITE'] = X['NAME_TYPE_SUITE'].fillna('Unknown')
        X['NAME_TYPE_SUITE'] = X['NAME_TYPE_SUITE'].astype('category')
        
        # AMT_GOODS_PRICE
        X['GOODS_PRICE_WAS_MISSING'] = X['AMT_GOODS_PRICE'].isnull().astype(float)
        X['AMT_GOODS_PRICE'] = X['AMT_GOODS_PRICE'].fillna(X['AMT_CREDIT'])
        
        # AMT_ANNUITY
        X['AMT_ANNUITY_WAS_MISSING'] = X['AMT_ANNUITY'].isnull().astype(float)
        X['AMT_ANNUITY'] = X['AMT_ANNUITY'].fillna(self.annuity_median_)
        
        # CNT_FAM_MEMBERS
        X['CNT_FAM_MEMBERS'] = X['CNT_FAM_MEMBERS'].fillna(self.fam_members_median_)
        
        # DAYS_LAST_PHONE_CHANGE
        X['DAYS_LAST_PHONE_CHANGE'] = X['DAYS_LAST_PHONE_CHANGE'].fillna(self.phone_change_median_)
        
        # ORGANISATION_TYPE
        if self.org_type_mode_ is not None:
            X['ORGANIZATION_TYPE'] = X['ORGANIZATION_TYPE'].fillna(self.org_type_mode_)
        
        return X


class CarAgeImputer(BaseEstimator, TransformerMixin):
    """Impute car age based on car ownership status"""
    
    def __init__(self):
        self.median_car_age_ = None
    
    def fit(self, X, y=None):
        # Calculate median car age for car owners
        self.median_car_age_ = X.loc[X['FLAG_OWN_CAR'] == 'Y', 'OWN_CAR_AGE'].median()
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Fill 0 for non-owners with missing car age
        X.loc[(X['FLAG_OWN_CAR'] == 'N') & X['OWN_CAR_AGE'].isna(), 'OWN_CAR_AGE'] = 0
        
        # Fill median for owners with missing car age
        X.loc[(X['FLAG_OWN_CAR'] == 'Y') & X['OWN_CAR_AGE'].isna(), 'OWN_CAR_AGE'] = self.median_car_age_
        
        return X


class EmploymentImputer(BaseEstimator, TransformerMixin):
    """Handle DAYS_EMPLOYED anomalies"""
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['DAYS_EMPLOYED'] = X['DAYS_EMPLOYED'].replace(365243, np.nan)
        X['DAYS_EMPLOYED'] = X['DAYS_EMPLOYED'].fillna(0)
        X['DAYS_EMPLOYED'] = pd.to_numeric(X['DAYS_EMPLOYED'], errors='coerce')
        return X


class OccupationImputer(BaseEstimator, TransformerMixin):
    """Impute occupation type based on income type and employment status"""
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        occ = X['OCCUPATION_TYPE'].copy()
        missing_mask = occ.isna() | (occ == '')
        no_employment_duration = X['DAYS_EMPLOYED'].isna() | (X['DAYS_EMPLOYED'] >= 0)
        
        # Conditions indicating likely unemployment
        # checked no raise value in .str bcoz of object dtype
        income_unemp = X['NAME_INCOME_TYPE'].str.contains(
            'Pensioner|Unemployed', case=False, regex=True, na=False
        )
        # update 2811 change the logic from 'or' to 'and' for the pensioner and has days_employed valid
        # Assign 'Unemployed' where appropriate
        occ.loc[missing_mask & (income_unemp & no_employment_duration)] = 'Unemployed'
        
        # Assign remaining missing as 'Laborers'
        occ.loc[missing_mask & ~(income_unemp & no_employment_duration)] = 'Laborers'
        
        X['OCCUPATION_TYPE'] = occ
        return X


class ExtSourceKNNImputer(BaseEstimator, TransformerMixin):
    """KNN Imputation for EXT_SOURCE columns using top correlated features"""
    
    def __init__(self, k=5, top_n=5):
        self.k = k
        self.top_n = top_n
        self.ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
        self.top_features_ = None
        self.knn_imputer_ = None
        self.knn_input_cols_ = None
    
    def fit(self, X, y=None):
        # Compute correlations
        corr = X.corr(numeric_only=True)[self.ext_cols]
        
        # Get top_n most correlated features per EXT column
        top_features = set()
        for col in self.ext_cols:
            top_corr = corr[col].drop(self.ext_cols, errors="ignore")
            top_n_feats = top_corr.abs().nlargest(self.top_n).index.tolist()
            top_features.update(top_n_feats)
        
        self.top_features_ = list(top_features)
        print(f'Selected top correlated features for KNN Imputer: {self.top_features_}')
        
        # Build KNN input columns
        self.knn_input_cols_ = self.ext_cols + self.top_features_
        
        # Fit KNN Imputer
        self.knn_imputer_ = KNNImputer(n_neighbors=self.k)
        knn_df = X[self.knn_input_cols_].copy()
        self.knn_imputer_.fit(knn_df)
        
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Apply KNN imputation
        knn_df = X[self.knn_input_cols_].copy()
        knn_arr = self.knn_imputer_.transform(knn_df)
        knn_df_imputed = pd.DataFrame(knn_arr, columns=self.knn_input_cols_, index=X.index)
        
        # Update only the EXT_SOURCE columns
        X[self.ext_cols] = knn_df_imputed[self.ext_cols]
        
        return X


class SimpleImputerTransformer(BaseEstimator, TransformerMixin):
    """Simple imputation (mean/median/constant) for specified columns"""
    
    def __init__(self, columns, strategy='median', fill_value=None):
        self.columns = columns
        self.strategy = strategy
        self.fill_value = fill_value
        self.imputer_ = None
    
    def fit(self, X, y=None):
        if self.strategy == 'constant':
            self.imputer_ = SimpleImputer(strategy=self.strategy, fill_value=self.fill_value)
        else:
            self.imputer_ = SimpleImputer(strategy=self.strategy)
        
        self.imputer_.fit(X[self.columns])
        return self
    
    def transform(self, X):
        X = X.copy()
        imputed_arr = self.imputer_.transform(X[self.columns])
        X[self.columns] = imputed_arr
        return X

# update 2811: create age and year employed features, drop days_birth and days_employed
class AgeFeatureCreator(BaseEstimator, TransformerMixin):
    """Create AGE feature from DAYS_BIRTH"""
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['AGE'] = (-X['DAYS_BIRTH'] / 365.25).astype(int)
        X = X.drop(columns=['DAYS_BIRTH'])
        return X

class YearEmployedFeatureCreator(BaseEstimator, TransformerMixin):
    """Create YEAR_EMPLOYED feature from DAYS_EMPLOYED"""
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['YEAR_EMPLOYED'] = (-X['DAYS_EMPLOYED'] / 365.25).astype(int)
        X = X.drop(columns=['DAYS_EMPLOYED'])
        return X

class HourBinner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        def bin_hour(x):
            if x < 6:
                return "Late_Night"
            elif x < 12:
                return "Morning"
            elif x < 18:
                return "Afternoon"
            else:
                return "Evening"
        
        X['HOUR_APPR_PROCESS_BIN'] = X['HOUR_APPR_PROCESS_START'].apply(bin_hour)
        
        return X

class CreditBureauProcessor(BaseEstimator, TransformerMixin):
    """Process credit bureau columns"""
    
    def __init__(self):
        self.credit_bureau_cols = [
            'AMT_REQ_CREDIT_BUREAU_HOUR',
            'AMT_REQ_CREDIT_BUREAU_DAY',
            'AMT_REQ_CREDIT_BUREAU_WEEK',
            'AMT_REQ_CREDIT_BUREAU_MON',
            'AMT_REQ_CREDIT_BUREAU_QRT',
            'AMT_REQ_CREDIT_BUREAU_YEAR'
        ]
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Create binary flag for missingness
        X['HAS_CREDIT_BUREAU_DATA'] = (~X['AMT_REQ_CREDIT_BUREAU_HOUR'].isna()).astype(int)
        
        # Fill 0 for all missing bureau counts
        X[self.credit_bureau_cols] = X[self.credit_bureau_cols].fillna(0)
        
        # Encode categorical
        X = self._encode_bureau_categorical(X)
        
        return X
    
    def _encode_bureau_categorical(self, df):
        for col in self.credit_bureau_cols:
            new_col = col + '_CAT'
            
            if col in ['AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY']:
                df[new_col] = (df[col] > 0).astype(int)
                df[new_col] = df[new_col].map({0: 'ZERO', 1: 'HAS_ENQUIRY'})
                
            elif col in ['AMT_REQ_CREDIT_BUREAU_WEEK']:
                df[new_col] = 'ZERO'
                df.loc[df[col] == 1, new_col] = 'ONE'
                df.loc[df[col] > 1, new_col] = 'MULTIPLE'
                
            elif col in ['AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT']:
                df[new_col] = 'ZERO'
                df.loc[df[col].between(1, 2), new_col] = 'LOW'
                df.loc[df[col] > 2, new_col] = 'HIGH'
                
            else:  # YEAR
                df[new_col] = 'ZERO'
                df.loc[df[col].between(1, 2), new_col] = 'LOW'
                df.loc[df[col].between(3, 5), new_col] = 'MEDIUM'
                df.loc[df[col] > 5, new_col] = 'HIGH'
            
            df = df.drop(col, axis=1)
        
        return df


class DocumentProcessor(BaseEstimator, TransformerMixin):
    """Process document flag columns"""
    
    def __init__(self):
        self.document_cols = [f'FLAG_DOCUMENT_{i}' for i in range(2, 22)]
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Create total documents submitted feature
        X['TOTAL_DOC_SUBMITTED'] = X[self.document_cols].sum(axis=1)
        
        # Drop individual document flags
        X = X.drop(columns=self.document_cols, errors='ignore')
        
        return X


class SocialCircleProcessor(BaseEstimator, TransformerMixin):
    """Process social circle columns with outlier capping"""
    
    def __init__(self):
        self.cols_social = [
            "OBS_30_CNT_SOCIAL_CIRCLE", "OBS_60_CNT_SOCIAL_CIRCLE",
            "DEF_30_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE"
        ]
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Cap outliers using IQR
        for col in self.cols_social:
            X[col + "_capped_iqr"] = self._cap_iqr(X[col])
        
        # Cap outliers using 99th percentile
        for col in self.cols_social:
            X[col + "_capped_p99"] = self._cap_percentile(X[col])
        
        return X
    
    def _cap_iqr(self, series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        upper = Q3 + 1.5 * IQR
        lower = Q1 - 1.5 * IQR
        return series.clip(lower, upper)
    
    def _cap_percentile(self, series, p=0.99):
        upper = series.quantile(p)
        return series.clip(upper=upper)


class AmountOutlierProcessor(BaseEstimator, TransformerMixin):
    """Process outliers in amount columns"""
    
    def __init__(self):
        self.amount_cols = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']
        self.caps_ = {}
    
    def fit(self, X, y=None):
        # Calculate 99.5th percentile thresholds
        for col in self.amount_cols:
            self.caps_[col] = np.percentile(X[col], 99.5)
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Winsorize and create outlier flags
        for col in self.amount_cols:
            threshold = self.caps_[col]
            X[f'{col}_outlier'] = (X[col] > threshold).astype(int)
            X[col] = X[col].clip(upper=threshold)
        
        return X


class CategoricalConverter(BaseEstimator, TransformerMixin):
    """Convert object dtype columns to category dtype"""
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        for col in X.select_dtypes(include='object').columns:
            X[col] = X[col].astype('category')
        return X


class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    """Label encode specified categorical columns"""
    
    def __init__(self, columns=None):
        """
        Parameters:
        -----------
        columns : list of str, optional
            List of column names to label encode. If None, will encode all categorical columns.
        """
        self.columns = columns
        self.label_encoders_ = {}
        self.columns_to_encode_ = None
    
    def fit(self, X, y=None):
        from sklearn.preprocessing import LabelEncoder
        
        X = X.copy()
        
        # Determine which columns to encode
        if self.columns is None:
            self.columns_to_encode_ = X.select_dtypes(include=['object', 'category']).columns.tolist()
        else:
            self.columns_to_encode_ = self.columns
        
        # Fit a label encoder for each column
        for col in self.columns_to_encode_:
            if col in X.columns:
                le = LabelEncoder()
                # Handle missing values by converting to string
                X[col] = X[col].astype(str)
                le.fit(X[col])
                self.label_encoders_[col] = le
        
        return self
    
    def transform(self, X):
        X = X.copy()
        
        for col in self.columns_to_encode_:
            if col in X.columns and col in self.label_encoders_:
                le = self.label_encoders_[col]
                X[col] = X[col].astype(str)
                
                # Handle unseen categories
                X[col] = X[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                X[col] = le.transform(X[col])
        
        return X


class CustomOrdinalEncoder(BaseEstimator, TransformerMixin):
    """Ordinal encode specified categorical columns with custom ordering"""
    
    def __init__(self, ordinal_mappings=None):
        """
        Parameters:
        -----------
        ordinal_mappings : dict, optional
            Dictionary mapping column names to ordered lists of categories.
            Example: {'education': ['Primary', 'Secondary', 'Higher']}
            If None, will use alphabetical ordering.
        """
        self.ordinal_mappings = ordinal_mappings or {}
        self.encoders_ = {}
        self.columns_to_encode_ = None
    
    def fit(self, X, y=None):
        from sklearn.preprocessing import OrdinalEncoder
        
        X = X.copy()
        self.columns_to_encode_ = list(self.ordinal_mappings.keys())
        
        for col in self.columns_to_encode_:
            if col in X.columns:
                # Create ordinal encoder with specified categories
                categories = [self.ordinal_mappings[col]]
                oe = OrdinalEncoder(categories=categories, handle_unknown='use_encoded_value', unknown_value=-1)
                oe.fit(X[[col]])
                self.encoders_[col] = oe
        
        return self
    
    def transform(self, X):
        X = X.copy()
        
        for col in self.columns_to_encode_:
            if col in X.columns and col in self.encoders_:
                oe = self.encoders_[col]
                X[col] = oe.transform(X[[col]])
        
        return X


class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    """One-hot encode specified categorical columns"""
    
    def __init__(self, columns=None, drop='first', max_categories=None, handle_unknown='ignore'):
        """
        Parameters:
        -----------
        columns : list of str, optional
            List of column names to one-hot encode. If None, will encode all categorical columns.
        drop : {'first', 'if_binary', None}, default='first'
            Whether to drop one of the categories per feature.
        max_categories : int, optional
            Maximum number of categories per column. Columns exceeding this will be skipped.
        handle_unknown : {'error', 'ignore'}, default='ignore'
            How to handle unknown categories during transform.
        """
        self.columns = columns
        self.drop = drop
        self.max_categories = max_categories
        self.handle_unknown = handle_unknown
        self.one_hot_encoder_ = None
        self.columns_to_encode_ = None
        self.feature_names_out_ = None
        self.columns_to_keep_ = None
    
    def fit(self, X, y=None):
        from sklearn.preprocessing import OneHotEncoder
        
        X = X.copy()
        
        # Determine which columns to encode
        if self.columns is None:
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        else:
            categorical_cols = self.columns
        
        # Filter by max_categories if specified
        self.columns_to_encode_ = []
        for col in categorical_cols:
            if col in X.columns:
                n_categories = X[col].nunique()
                if self.max_categories is None or n_categories <= self.max_categories:
                    self.columns_to_encode_.append(col)
                else:
                    print(f"Skipping {col}: {n_categories} categories exceeds max_categories={self.max_categories}")
        
        # Store columns that won't be encoded
        self.columns_to_keep_ = [col for col in X.columns if col not in self.columns_to_encode_]
        
        if len(self.columns_to_encode_) > 0:
            # Fit one-hot encoder
            self.one_hot_encoder_ = OneHotEncoder(
                drop=self.drop,
                sparse_output=False,
                handle_unknown=self.handle_unknown
            )
            self.one_hot_encoder_.fit(X[self.columns_to_encode_])
            
            # Get feature names
            self.feature_names_out_ = self.one_hot_encoder_.get_feature_names_out(self.columns_to_encode_)
        
        return self
    
    def transform(self, X):
        X = X.copy()
        
        if len(self.columns_to_encode_) > 0 and self.one_hot_encoder_ is not None:
            # Transform categorical columns
            encoded_array = self.one_hot_encoder_.transform(X[self.columns_to_encode_])
            encoded_df = pd.DataFrame(
                encoded_array,
                columns=self.feature_names_out_,
                index=X.index
            )
            
            # Combine with non-encoded columns
            X_non_encoded = X[self.columns_to_keep_]
            X_transformed = pd.concat([X_non_encoded, encoded_df], axis=1)
            
            return X_transformed
        else:
            return X


class SmartCategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Intelligently encode categorical variables based on cardinality:
    - Binary columns: Label encode
    - Low cardinality (<=10): One-hot encode
    - High cardinality (>10): Target/Frequency encode or keep as-is
    """
    
    def __init__(self, low_cardinality_threshold=10, encoding_strategy='auto'):
        """
        Parameters:
        -----------
        low_cardinality_threshold : int, default=10
            Threshold for determining low vs high cardinality
        encoding_strategy : {'auto', 'frequency', 'target'}, default='auto'
            Strategy for high cardinality features
        """
        self.low_cardinality_threshold = low_cardinality_threshold
        self.encoding_strategy = encoding_strategy
        self.binary_cols_ = []
        self.low_card_cols_ = []
        self.high_card_cols_ = []
        self.label_encoders_ = {}
        self.one_hot_encoder_ = None
        self.frequency_maps_ = {}
        self.columns_to_keep_ = None
        self.feature_names_out_ = None
    
    def fit(self, X, y=None):
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder
        
        X = X.copy()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Categorize columns by cardinality
        for col in categorical_cols:
            n_unique = X[col].nunique()
            
            if n_unique == 2:
                self.binary_cols_.append(col)
            elif n_unique <= self.low_cardinality_threshold:
                self.low_card_cols_.append(col)
            else:
                self.high_card_cols_.append(col)
        
        print(f"Binary columns ({len(self.binary_cols_)}): {self.binary_cols_}")
        print(f"Low cardinality columns ({len(self.low_card_cols_)}): {self.low_card_cols_}")
        print(f"High cardinality columns ({len(self.high_card_cols_)}): {self.high_card_cols_}")
        
        # Fit label encoders for binary columns
        for col in self.binary_cols_:
            le = LabelEncoder()
            X[col] = X[col].astype(str)
            le.fit(X[col])
            self.label_encoders_[col] = le
        
        # Fit one-hot encoder for low cardinality columns
        if len(self.low_card_cols_) > 0:
            self.one_hot_encoder_ = OneHotEncoder(
                drop='first',
                sparse_output=False,
                handle_unknown='ignore'
            )
            self.one_hot_encoder_.fit(X[self.low_card_cols_])
            self.feature_names_out_ = self.one_hot_encoder_.get_feature_names_out(self.low_card_cols_)
        
        # Fit frequency encoding for high cardinality columns
        for col in self.high_card_cols_:
            freq_map = X[col].value_counts(normalize=True).to_dict()
            self.frequency_maps_[col] = freq_map
        
        # Store columns to keep as-is
        non_categorical = [col for col in X.columns if col not in categorical_cols]
        self.columns_to_keep_ = non_categorical
        
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Label encode binary columns
        for col in self.binary_cols_:
            if col in X.columns:
                le = self.label_encoders_[col]
                X[col] = X[col].astype(str)
                X[col] = X[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                X[col] = le.transform(X[col])
        
        # One-hot encode low cardinality columns
        encoded_dfs = []
        if len(self.low_card_cols_) > 0 and self.one_hot_encoder_ is not None:
            encoded_array = self.one_hot_encoder_.transform(X[self.low_card_cols_])
            encoded_df = pd.DataFrame(
                encoded_array,
                columns=self.feature_names_out_,
                index=X.index
            )
            encoded_dfs.append(encoded_df)
        
        # Frequency encode high cardinality columns
        for col in self.high_card_cols_:
            if col in X.columns:
                freq_map = self.frequency_maps_[col]
                X[col + '_freq'] = X[col].map(freq_map).fillna(0)
        
        # Combine all features
        result_dfs = [X[self.columns_to_keep_]]
        
        # Add binary encoded columns
        for col in self.binary_cols_:
            if col in X.columns:
                result_dfs.append(X[[col]])
        
        # Add one-hot encoded columns
        if encoded_dfs:
            result_dfs.extend(encoded_dfs)
        
        # Add frequency encoded columns
        for col in self.high_card_cols_:
            if col + '_freq' in X.columns:
                result_dfs.append(X[[col + '_freq']])
        
        X_transformed = pd.concat(result_dfs, axis=1)
        
        return X_transformed


class FlexibleCategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Flexible encoder that allows you to specify which columns get which encoding type.
    """
    
    def __init__(self, 
                 label_encode_cols=None,
                 onehot_encode_cols=None,
                 ordinal_encode_cols=None,
                 ordinal_mappings=None,
                 frequency_encode_cols=None,
                 onehot_drop='first',
                 handle_unknown='ignore'):
        """
        Parameters:
        -----------
        label_encode_cols : list of str, optional
            Columns to label encode
        onehot_encode_cols : list of str, optional
            Columns to one-hot encode
        ordinal_encode_cols : list of str, optional
            Columns to ordinal encode (must provide ordinal_mappings)
        ordinal_mappings : dict, optional
            Dictionary mapping column names to ordered category lists
            Example: {'education': ['Primary', 'Secondary', 'Higher']}
        frequency_encode_cols : list of str, optional
            Columns to frequency encode
        onehot_drop : {'first', 'if_binary', None}, default='first'
            Whether to drop one category in one-hot encoding
        handle_unknown : {'error', 'ignore'}, default='ignore'
            How to handle unknown categories
        """
        self.label_encode_cols = label_encode_cols or []
        self.onehot_encode_cols = onehot_encode_cols or []
        self.ordinal_encode_cols = ordinal_encode_cols or []
        self.ordinal_mappings = ordinal_mappings or {}
        self.frequency_encode_cols = frequency_encode_cols or []
        self.onehot_drop = onehot_drop
        self.handle_unknown = handle_unknown
        
        # Fitted attributes
        self.label_encoders_ = {}
        self.onehot_encoder_ = None
        self.ordinal_encoders_ = {}
        self.frequency_maps_ = {}
        self.onehot_feature_names_ = None
        self.columns_to_keep_ = None
    
    def fit(self, X, y=None):
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
        
        X = X.copy()
        
        # Determine columns to keep unchanged
        all_encoded_cols = (self.label_encode_cols + self.onehot_encode_cols + 
                           self.ordinal_encode_cols + self.frequency_encode_cols)
        self.columns_to_keep_ = [col for col in X.columns if col not in all_encoded_cols]
        
        # 1. Fit label encoders
        for col in self.label_encode_cols:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = X[col].astype(str)
                le.fit(X[col])
                self.label_encoders_[col] = le
                print(f"Label encoding: {col} ({X[col].nunique()} categories)")
        
        # 2. Fit one-hot encoder
        if len(self.onehot_encode_cols) > 0:
            valid_onehot_cols = [col for col in self.onehot_encode_cols if col in X.columns]
            if valid_onehot_cols:
                self.onehot_encoder_ = OneHotEncoder(
                    drop=self.onehot_drop,
                    sparse_output=False,
                    handle_unknown=self.handle_unknown
                )
                self.onehot_encoder_.fit(X[valid_onehot_cols])
                self.onehot_feature_names_ = self.onehot_encoder_.get_feature_names_out(valid_onehot_cols)
                for col in valid_onehot_cols:
                    print(f"One-hot encoding: {col} ({X[col].nunique()} categories)")
        
        # 3. Fit ordinal encoders
        for col in self.ordinal_encode_cols:
            if col in X.columns and col in self.ordinal_mappings:
                categories = [self.ordinal_mappings[col]]
                oe = OrdinalEncoder(
                    categories=categories,
                    handle_unknown='use_encoded_value',
                    unknown_value=-1
                )
                oe.fit(X[[col]])
                self.ordinal_encoders_[col] = oe
                #print(f"Ordinal encoding: {col} with order {self.ordinal_mappings[col]}")
        
        # 4. Fit frequency encoders
        for col in self.frequency_encode_cols:
            if col in X.columns:
                freq_map = X[col].value_counts(normalize=True).to_dict()
                #print(f'frequency map for {col}: {freq_map}')
                self.frequency_maps_[col] = freq_map
                #print(f"Frequency encoding: {col} ({X[col].nunique()} categories)")
        
        return self
    
    def transform(self, X):
        X = X.copy()
        result_dfs = []
        
        # Keep unchanged columns
        if self.columns_to_keep_:
            result_dfs.append(X[self.columns_to_keep_])
        
        # 1. Label encode
        for col in self.label_encode_cols:
            if col in X.columns and col in self.label_encoders_:
                le = self.label_encoders_[col]
                X[col] = X[col].astype(str)
                # Handle unseen categories
                X[col] = X[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                X[col] = le.transform(X[col])
                result_dfs.append(X[[col]])
        
        # 2. One-hot encode
        if self.onehot_encoder_ is not None:
            valid_onehot_cols = [col for col in self.onehot_encode_cols if col in X.columns]
            if valid_onehot_cols:
                encoded_array = self.onehot_encoder_.transform(X[valid_onehot_cols])
                encoded_df = pd.DataFrame(
                    encoded_array,
                    columns=self.onehot_feature_names_,
                    index=X.index
                )
                result_dfs.append(encoded_df)
        
        # 3. Ordinal encode
        for col in self.ordinal_encode_cols:
            if col in X.columns and col in self.ordinal_encoders_:
                oe = self.ordinal_encoders_[col]
                ordinal_encoded = oe.transform(X[[col]])
                result_dfs.append(pd.DataFrame(ordinal_encoded, columns=[col], index=X.index))
        
        # 4. Frequency encode
        for col in self.frequency_encode_cols:
            if col in X.columns and col in self.frequency_maps_:
                # print('Done create freq map, start transform')
                freq_map = self.frequency_maps_[col]
                # print(f'freq map for {col} during transform: {freq_map}')
                freq_encoded = X[col].map(freq_map).fillna(0)
                # print(f'freq encoded sample for {col}: {freq_encoded.head()}')
                freq_df = freq_encoded.rename(col + "_freq").to_frame()
                #print(freq_df.head())
                result_dfs.append(freq_df)

                X = X.drop(columns=[col])
        # Combine all encoded features
        X_transformed = pd.concat(result_dfs, axis=1)
        
        return X_transformed


# Usage example:
def create_preprocessing_pipeline(encoding_type='smart', encoding_config=None):
    """
    Create the full preprocessing pipeline
    
    Parameters:
    -----------
    encoding_type : {'smart', 'flexible', 'onehot', 'label', 'ordinal', 'none'}
        Type of categorical encoding to use
    encoding_config : dict, optional
        Configuration for 'flexible' encoding type. Example:
        {
            'label_encode_cols': ['CODE_GENDER', 'FLAG_OWN_CAR'],
            'onehot_encode_cols': ['NAME_CONTRACT_TYPE', 'NAME_INCOME_TYPE'],
            'ordinal_encode_cols': ['NAME_EDUCATION_TYPE'],
            'ordinal_mappings': {
                'NAME_EDUCATION_TYPE': ['Lower secondary', 'Secondary / secondary special',
                                       'Incomplete higher', 'Higher education']
            },
            'frequency_encode_cols': ['ORGANIZATION_TYPE']
        }
    """
    from sklearn.pipeline import Pipeline
    
    # Define social and ext columns
    cols_social = [
        "OBS_30_CNT_SOCIAL_CIRCLE", "OBS_60_CNT_SOCIAL_CIRCLE",
        "DEF_30_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE"
    ]
    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    
    # Base pipeline steps
    steps = [
        ('basic_imputer', BasicImputerTransformer()),
        ('car_age_imputer', CarAgeImputer()),
        ('employment_imputer', EmploymentImputer()),
        ('occupation_imputer', OccupationImputer()),
        # Choose either KNN or Simple imputer for EXT_SOURCE
        # ('ext_knn_imputer', ExtSourceKNNImputer(k=5, top_n=5)),
        ('ext_simple_imputer', SimpleImputerTransformer(columns=ext_cols, strategy='median')),
        ('social_simple_imputer', SimpleImputerTransformer(columns=cols_social, strategy='median')),
        ('age_creator', AgeFeatureCreator()),
        ('year_employed_creator', YearEmployedFeatureCreator()),
        ('credit_bureau_processor', CreditBureauProcessor()),
        ('document_processor', DocumentProcessor()),
        # ('social_outlier_processor', SocialCircleProcessor()),
        # ('amount_outlier_processor', AmountOutlierProcessor()),
    ]
    
    # Add encoding step based on type
    if encoding_type == 'smart':
        steps.append(('smart_encoder', SmartCategoricalEncoder(low_cardinality_threshold=10)))
    
    elif encoding_type == 'flexible':
        if encoding_config is None:
            raise ValueError("encoding_config must be provided when using 'flexible' encoding_type")
        steps.append(('flexible_encoder', FlexibleCategoricalEncoder(**encoding_config)))
    
    elif encoding_type == 'onehot':
        # One-hot encode all categorical columns with max 15 categories
        steps.append(('onehot_encoder', CustomOneHotEncoder(max_categories=15, drop='first')))
    
    elif encoding_type == 'label':
        # Label encode all categorical columns
        steps.append(('label_encoder', CustomLabelEncoder()))
    
    elif encoding_type == 'ordinal':
        # Example ordinal mappings - customize based on your domain knowledge
        ordinal_mappings = {
            'NAME_EDUCATION_TYPE': [
                'Lower secondary', 'Secondary / secondary special',
                'Incomplete higher', 'Higher education'
            ],
            # Add more ordinal mappings as needed
        }
        steps.append(('ordinal_encoder', CustomOrdinalEncoder(ordinal_mappings=ordinal_mappings)))
    
    elif encoding_type == 'none':
        # Just convert to category dtype
        steps.append(('categorical_converter', CategoricalConverter()))
    
    pipeline = Pipeline(steps)
    return pipeline

def save_preprocessing_pipeline(pipeline, filepath: str):
    """
    Save preprocessing pipeline to disk
    
    Parameters:
    -----------
    pipeline : sklearn.pipeline.Pipeline
        Preprocessing pipeline to save
    filepath : str
        Path to save pipeline
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, filepath)
    print(f"Preprocessing pipeline saved to {filepath}")



# Example usage with different encoders:
if __name__ == "__main__":
    # ========== EXAMPLE 1: Smart encoding (automatic) ==========
    # pipeline = create_preprocessing_pipeline(encoding_type='smart')
    # df_transformed = pipeline.fit_transform(df)
    
    # ========== EXAMPLE 2: Flexible encoding (YOU DEFINE EACH COLUMN) ==========
    # This is what you requested - full control over which columns get which encoding
    encoding_config = {
        'label_encode_cols': [
            'CODE_GENDER',           # Binary: M/F
            'FLAG_OWN_CAR',          # Binary: Y/N
            'FLAG_OWN_REALTY'        # Binary: Y/N
        ],
        'onehot_encode_cols': [
            'NAME_CONTRACT_TYPE',    # Low cardinality: Cash loans, Revolving loans
            'NAME_INCOME_TYPE',      # Low cardinality: ~8 types
            'NAME_FAMILY_STATUS',    # Low cardinality: ~6 types
            'NAME_HOUSING_TYPE', # 7 type
            'NAME_TYPE_SUITE',
            'WEEKDAY_APPR_PROCESS_START',
            'AMT_REQ_CREDIT_BUREAU_HOUR_CAT', 
            'AMT_REQ_CREDIT_BUREAU_DAY_CAT',      
            'AMT_REQ_CREDIT_BUREAU_WEEK_CAT', 
            'AMT_REQ_CREDIT_BUREAU_MON_CAT',      
            'AMT_REQ_CREDIT_BUREAU_QRT_CAT', 
            'AMT_REQ_CREDIT_BUREAU_YEAR_CAT'    
        ],
        'ordinal_encode_cols': [
            'NAME_EDUCATION_TYPE'    # Has natural ordering
        ],
        'ordinal_mappings': {
            'NAME_EDUCATION_TYPE': [
                'Lower secondary',
                'Secondary / secondary special',
                'Incomplete higher',
                'Higher education',
                'Academic degree'
            ]
        },
        'frequency_encode_cols': [
            'ORGANIZATION_TYPE',     # High cardinality: 50+ types
            'OCCUPATION_TYPE'        # Medium-high cardinality: ~18 types
        ]
    }
    
    pipeline = create_preprocessing_pipeline(
        encoding_type='flexible',
        encoding_config=encoding_config
    )

    save_preprocessing_pipeline(pipeline, 'models/preprocessing_pipeline.pkl')
    # ========== EXAMPLE 3: Mix encoders in custom pipeline ==========
    # from sklearn.pipeline import Pipeline
    # custom_pipeline = Pipeline([
    #     ('preprocessor', create_preprocessing_pipeline(encoding_type='none')),
    #     ('label_encoder', CustomLabelEncoder(columns=['CODE_GENDER', 'FLAG_OWN_CAR'])),
    #     ('onehot_encoder', CustomOneHotEncoder(columns=['NAME_CONTRACT_TYPE', 'NAME_INCOME_TYPE'])),
    # ])
    
    