import pandas as pd
import numpy as np
import warnings
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
def apply_processing_steps(df):
    # drop columns respectively with more than threshold% missing values after EDA
    columns_to_drop = ['APARTMENTS_AVG',
    'BASEMENTAREA_AVG',
    'YEARS_BEGINEXPLUATATION_AVG',
    'YEARS_BUILD_AVG',
    'COMMONAREA_AVG',
    'ELEVATORS_AVG',
    'ENTRANCES_AVG',
    'FLOORSMAX_AVG',
    'FLOORSMIN_AVG',
    'LANDAREA_AVG',
    'LIVINGAPARTMENTS_AVG',
    'LIVINGAREA_AVG',
    'NONLIVINGAPARTMENTS_AVG',
    'NONLIVINGAREA_AVG',
    'APARTMENTS_MODE',
    'BASEMENTAREA_MODE',
    'YEARS_BEGINEXPLUATATION_MODE',
    'YEARS_BUILD_MODE',
    'COMMONAREA_MODE',
    'ELEVATORS_MODE',
    'ENTRANCES_MODE',
    'FLOORSMAX_MODE',
    'FLOORSMIN_MODE',
    'LANDAREA_MODE',
    'LIVINGAPARTMENTS_MODE',
    'LIVINGAREA_MODE',
    'NONLIVINGAPARTMENTS_MODE',
    'NONLIVINGAREA_MODE',
    'APARTMENTS_MEDI',
    'BASEMENTAREA_MEDI',
    'YEARS_BEGINEXPLUATATION_MEDI',
    'YEARS_BUILD_MEDI',
    'COMMONAREA_MEDI',
    'ELEVATORS_MEDI',
    'ENTRANCES_MEDI',
    'FLOORSMAX_MEDI',
    'FLOORSMIN_MEDI',
    'LANDAREA_MEDI',
    'LIVINGAPARTMENTS_MEDI',
    'LIVINGAREA_MEDI',
    'NONLIVINGAPARTMENTS_MEDI',
    'NONLIVINGAREA_MEDI',
    'FONDKAPREMONT_MODE' ,
    'HOUSETYPE_MODE' ,
    'TOTALAREA_MODE' ,
    'WALLSMATERIAL_MODE' ,
    'EMERGENCYSTATE_MODE' ]
    # print(f'Drop {columns_to_drop.__len__()} house columns with more than threshold% missing values')
    df_prep = df.copy()
    df_prep = df_prep.drop(columns=columns_to_drop, errors = 'ignore')

    #1 NAME_TYPE_SUITE - missing 882 values(MAR))
    df_prep['NAME_TYPE_SUITE'] = df_prep['NAME_TYPE_SUITE'].fillna('Unknown') 
    df_prep['NAME_TYPE_SUITE'] = df_prep['NAME_TYPE_SUITE'].astype('category')

    # 2. AMT_GOODS_PRICE – missing(MCAR) 174 values
    df_prep['GOODS_PRICE_WAS_MISSING'] = df_prep['AMT_GOODS_PRICE'].isnull().astype(float)
    df_prep['AMT_GOODS_PRICE'] = df_prep['AMT_GOODS_PRICE'].fillna(df_prep['AMT_CREDIT'])

    # 3. AMT_ANNUITY – only 9 missing values(MCAR) → fill in the median to avoid lossing of important feature data
    df_prep['AMT_ANNUITY_WAS_MISSING'] = df_prep['AMT_ANNUITY'].isnull().astype(float)
    df_prep['AMT_ANNUITY'] = df_prep['AMT_ANNUITY'].fillna(df_prep['AMT_ANNUITY'].median())

    # XỬ LÝ nan CODE_GENDER
    print(f"Original dataset size: {len(df_prep):,} rows")
    xna_gender_count = (df_prep['CODE_GENDER'] == 'XNA').sum()
    # print(f"Rows with CODE_GENDER == 'XNA': {xna_gender_count:,} (will be dropped)")
    # df_prep = df_prep[df_prep['CODE_GENDER'] != 'XNA']  

    # XỬ LÝ nan cho OWN_CAR_AGE

    non_owners_missing = ((df_prep['FLAG_OWN_CAR'] == 'N') & df_prep['OWN_CAR_AGE'].isna()).sum()
    owners_missing = ((df_prep['FLAG_OWN_CAR'] == 'Y') & df_prep['OWN_CAR_AGE'].isna()).sum()

    # Fill 0 cho bọn không có xe với độ tuổi xe bị thiếu
    df_prep.loc[(df_prep['FLAG_OWN_CAR'] == 'N') & df_prep['OWN_CAR_AGE'].isna(), 'OWN_CAR_AGE'] = 0
    median_car_age = df_prep.loc[df_prep['FLAG_OWN_CAR'] == 'Y', 'OWN_CAR_AGE'].median()

    df_prep.loc[
        (df_prep['FLAG_OWN_CAR'] == 'Y') & (df_prep['OWN_CAR_AGE'].isna()),
        'OWN_CAR_AGE'
    ] = median_car_age 

    # XỬ LÝ DAYS_EMPLOYED: 
    df_prep['DAYS_EMPLOYED'] = df_prep['DAYS_EMPLOYED'].replace(365243, np.nan)
    df_prep['DAYS_EMPLOYED'] = df_prep['DAYS_EMPLOYED'].fillna(0)

    # XỬ LÝ OCCUPATION_TYPE
    def impute_occupation(df):
        occ = df['OCCUPATION_TYPE'].copy()
        missing_mask = occ.isna() | (occ == '')
        # Conditions indicating likely unemployment / no defined occupation
        income_unemp = df['NAME_INCOME_TYPE'].str.contains('Pensioner|Unemployed', case=False, regex=True, na=False)
        no_employment_duration = df['DAYS_EMPLOYED'].isna() | (df['DAYS_EMPLOYED'] >= 0)  # người missing employment duration hoặc không có ngày làm việc hợp lệ
        # Cho thành 'Unemployed' nơi cả occupation missing và đang trong độ tuổi không có nghề nghiệp
        occ.loc[missing_mask & (income_unemp | no_employment_duration)] = 'Unemployed'
        # Những cái còn lại gán thành 'Laborers'
        occ.loc[missing_mask & ~(income_unemp | no_employment_duration)] = 'Laborers'
        return occ

    df_prep['OCCUPATION_TYPE'] = impute_occupation(df_prep)

    # XỬ LÝ CNT_FAM_MEMBERS và DAYS_LAST_PHONE_CHANGE
    # 3. CNT_FAM_MEMBERS median imputation 
    df_prep['CNT_FAM_MEMBERS'] = df_prep['CNT_FAM_MEMBERS'].fillna(df_prep['CNT_FAM_MEMBERS'].median())

    # 4. DAYS_LAST_PHONE_CHANGE median imputation
    df_prep['DAYS_LAST_PHONE_CHANGE'] = df_prep['DAYS_LAST_PHONE_CHANGE'].fillna(df_prep['DAYS_LAST_PHONE_CHANGE'].median())

    # fillna ORGANIZATION_TYPE with mode
    df_prep['ORGANISATION_TYPE'] = df['ORGANIZATION_TYPE'].fillna(df['ORGANIZATION_TYPE'].mode()[0])



    df_prep['AGE'] = (-df_prep['DAYS_BIRTH'] / 365.25).astype(int)
    df_prep.drop(columns=['DAYS_BIRTH'], inplace=True)

    # process ext features
    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]

    df_ext = df[ext_cols].copy()

    # BƯỚC XỬ LÍ MISSING

    # 1. Dùng Simpleimputer( mean/median/0)
    #mean/median
    def simple_impute(df, strategy="mean"):
        imp = SimpleImputer(strategy=strategy)
        arr = imp.fit_transform(df)
        return pd.DataFrame(arr, columns=[c + f"_simple_{strategy}" for c in df.columns])

    # df_ext_mean = simple_impute(df_ext, strategy="mean")
    # df_ext_median = simple_impute(df_ext, strategy="median")

    # constant = 0
    def simple_impute_zero(df):
        imp = SimpleImputer(strategy="constant", fill_value=0)
        arr = imp.fit_transform(df)
        return pd.DataFrame(arr, columns=[c + "_simple_0" for c in df.columns])

    # df_ext_zero = simple_impute_zero(df_ext)

    # 2. Dùng KNN
    def impute_ext_knn(df, k=5, top_n=5):
        """
        Impute EXT_SOURCE_1/2/3 using KNNImputer.
        - Selects top_n features with highest absolute correlation
        with the EXT_SOURCE columns.
        - Uses ONLY those features + EXT columns to fit KNN.
        Returns a DataFrame of the 3 imputed EXT_SOURCE_* columns.
        """

        ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
        ext = df[ext_cols].copy()

        # ==== 1. Compute correlations ====
        corr = df.corr(numeric_only=True)[ext_cols]

        # ==== 2. Get top_n most correlated features per EXT column ====
        top_features = set()  # use set to avoid duplicates
        for col in ext_cols:
            # drop the EXT_SOURCE columns to avoid trivial self-correlation
            top_corr = corr[col].drop(ext_cols, errors="ignore")
            # pick highest absolute correlations
            top_n_feats = top_corr.abs().nlargest(top_n).index.tolist()
            top_features.update(top_n_feats)

        top_features = list(top_features)
        print(f'The selected top correlated features for KNN Imputer: {top_features}')

        # ==== 3. Build KNN input data ====
        # Use EXT_SOURCE columns + selected correlated features
        knn_input_cols = ext_cols + top_features
        knn_df = df[knn_input_cols].copy()

        # ==== 4. Apply KNN Imputer ====
        knn = KNNImputer(n_neighbors=k)
        knn_arr = knn.fit_transform(knn_df)

        knn_df_imputed = pd.DataFrame(knn_arr, columns=knn_input_cols, index=df.index)

        # ==== 5. Return only the imputed EXT columns ====
        return knn_df_imputed[ext_cols]


    # df_ext_knn =impute_ext_knn(df_ext)

    # BƯỚC GỘP THÀNH 1 FEATURE
    #  1. Mean score 
    def combine_mean(df):
        df_out = df.copy()
        df_out["EXT_SOURCE_MEAN"] = df.mean(axis=1)
        return df_out[["EXT_SOURCE_MEAN"]]

    #  2. PCA 1 component 
    def combine_pca(df):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df)

        pca = PCA(n_components=1)
        comp = pca.fit_transform(X_scaled)

        return pd.DataFrame(comp, columns=["EXT_SOURCE_PCA1"])

    #  3. weighted average 
    def combine_weight(df, w1=0.2, w2=0.5, w3=0.3):
        weights = np.array([w1, w2, w3])
        new_col = df.values.dot(weights)
        return pd.DataFrame(new_col, columns=["EXT_SOURCE_WEIGHTED"])

    cols_social = [
        "OBS_30_CNT_SOCIAL_CIRCLE", "OBS_60_CNT_SOCIAL_CIRCLE",
        "DEF_30_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE"
    ]

    df_social = df[cols_social].copy()   


    #treat ext columns and social columns by simple imputation
    df_prep[ext_cols] = simple_impute(df_ext, strategy="median")
    df_prep[cols_social] = simple_impute(df_social, strategy="median")

    document_cols = [f'FLAG_DOCUMENT_{i}' for i in range(2, 22)]
    credit_bureau_cols = [
        'AMT_REQ_CREDIT_BUREAU_HOUR',
        'AMT_REQ_CREDIT_BUREAU_DAY',
        'AMT_REQ_CREDIT_BUREAU_WEEK',
        'AMT_REQ_CREDIT_BUREAU_MON',
        'AMT_REQ_CREDIT_BUREAU_QRT',
        'AMT_REQ_CREDIT_BUREAU_YEAR'
        ]
    # Feature 1: Binary flag for missingness
    df_prep['HAS_CREDIT_BUREAU_DATA'] = (~df_prep['AMT_REQ_CREDIT_BUREAU_HOUR'].isna()).astype(int)
            
    # Feature 2: Fill 0 for all bureau counts
    df_prep[credit_bureau_cols] = df_prep[credit_bureau_cols].fillna(0) 
    
    # Feature 1: Binary flag for missingness
    df_prep['HAS_CREDIT_BUREAU_DATA'] = (~df_prep['AMT_REQ_CREDIT_BUREAU_HOUR'].isna()).astype(int)
            
    # Feature 2: Fill 0 for all bureau counts
    df_prep[credit_bureau_cols] = df_prep[credit_bureau_cols].fillna(0)

    df_prep['TOTAL_DOC_SUBMITTED'] = df[document_cols].sum(axis=1)
    drop_cols = [col for col in df_prep.columns if col.startswith('FLAG_DOCUMENT_')]
    df_prep = df_prep.drop(columns=drop_cols)

    def encode_bureau_categorical(df, credit_bureau_cols):
        
        for col in credit_bureau_cols:
            new_col = col + '_CAT'
            
            # Define bins dựa trên distribution
            if col in ['AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY']:
                # HOUR/DAY: 0 vs >0
                df[new_col] = (df[col] > 0).astype(int)
                df[new_col] = df[new_col].map({0: 'ZERO', 1: 'HAS_ENQUIRY'})
                
            elif col in ['AMT_REQ_CREDIT_BUREAU_WEEK']:
                # WEEK: 0, 1, >1
                df[new_col] = 'ZERO'
                df.loc[df[col] == 1, new_col] = 'ONE'
                df.loc[df[col] > 1, new_col] = 'MULTIPLE'
                
            elif col in ['AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT']:
                # MONTH/QRT: 0, 1-2, >2
                df[new_col] = 'ZERO'
                df.loc[df[col].between(1, 2), new_col] = 'LOW'
                df.loc[df[col] > 2, new_col] = 'HIGH'
                
            else:  # YEAR
                # YEAR: 0, 1-2, 3-5, >5
                df[new_col] = 'ZERO'
                df.loc[df[col].between(1, 2), new_col] = 'LOW'
                df.loc[df[col].between(3, 5), new_col] = 'MEDIUM'
                df.loc[df[col] > 5, new_col] = 'HIGH'

            df = df.drop(col, axis=1)
        
        return df               # clip: loại bỏ những giá trị cực đoan
    df_prep = encode_bureau_categorical(df_prep, credit_bureau_cols)

    #change object dtype to category dtype
    for col in df_prep.select_dtypes(include='object').columns:
        df_prep[col] = df_prep[col].astype('category')

    # 1. HÀM CHECK & CAP OUTLIER BẰNG IQR
    def cap_iqr(series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        upper = Q3 + 1.5 * IQR
        lower = Q1 - 1.5 * IQR
        return series.clip(lower, upper)
    def cap_percentile(series, p=0.99):
        upper = series.quantile(p)
        return series.clip(upper=upper)
    def process_outlier(df_prep):
        for col in cols_social:
            df_prep[col + "_capped_iqr"] = cap_iqr(df_prep[col])

        for col in cols_social:
            df_prep[col + "_capped_p99"] = cap_percentile(df_prep[col])
        
        # 1. Dùng threshold 
        caps = {
            'AMT_INCOME_TOTAL' : np.percentile(df_prep['AMT_INCOME_TOTAL'], 99.5),   
            'AMT_CREDIT'       : np.percentile(df_prep['AMT_CREDIT'], 99.5),        
            'AMT_ANNUITY'      : np.percentile(df_prep['AMT_ANNUITY'], 99.5),       
            'AMT_GOODS_PRICE'  : np.percentile(df_prep['AMT_GOODS_PRICE'], 99.5),   
        }

        # 2. Winsorize (cắt ngọn)(mọi giá trị lớn hơn threshold bị ép xuống bằng threshold) + tạo flag outlier 
        for col, threshold in caps.items():
            df_prep[f'{col}_outlier'] = (df_prep[col] > threshold).astype(int)   # flag: giữ lại thông tin nguwofi này là từng là cực giàu/ vay cực lớn
            df_prep[col] = df_prep[col].clip(upper=threshold) 
                


