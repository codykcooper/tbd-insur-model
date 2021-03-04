```python
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

from imblearn.over_sampling import SMOTE 

from pandas_profiling import ProfileReport
```

# EDA Tools


```python
df=pd.read_csv('data/travel_insurance.csv')
df['claim_num'] = [0 if x == 'No' else 1 for x in df['claim']]
```

### Pandas profiling is being used to automatically generate exploratory analysis of the data set. 

Main goals of the report are to identify:

- Any missing or unusual characteristics (e.g., skew, low incidence, etc.) in the data
- Get a sense of correlations between variables (both features and outcomes)
- miscelaneous data issues


```python
profile = ProfileReport(df, title='tbd Insur Profiling Report', explorative=True)
```

The below code will save the exploratory data analysis to an html file that can be evaluated for issues. 


```python
profile.to_file("tbd-insur eda output.html")
;
```


    HBox(children=(HTML(value='Summarize dataset'), FloatProgress(value=0.0, max=25.0), HTML(value='')))


    



    HBox(children=(HTML(value='Generate report structure'), FloatProgress(value=0.0, max=1.0), HTML(value='')))


    



    HBox(children=(HTML(value='Render HTML'), FloatProgress(value=0.0, max=1.0), HTML(value='')))


    



    HBox(children=(HTML(value='Export report to file'), FloatProgress(value=0.0, max=1.0), HTML(value='')))


    





    ''



### Based on assessment of the report there are several issues that need to be addressed before modeling:
- High missingness in Gender variable
- High cardinality in destination variable. For simplicity, will likely bin to high medium low destination categories.
- Duration is highly skewed and contains missing values

If interested in this assessment, you can view report output in "tbd-insur eda output.html" in this repo

## Cleaning Data for modeling



### Cleaning gender column
There are three categories in the gender column, male, female, and missing. 
There may be useful information in this column, so we'll create two columns, one for male and one for female, to compare against missing in the final model. 


```python
df=pd.concat([df,pd.get_dummies(df['gender'])], axis = 1)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>agency</th>
      <th>agency_type</th>
      <th>distribution_channel</th>
      <th>product_name</th>
      <th>claim</th>
      <th>duration</th>
      <th>destination</th>
      <th>net_sales</th>
      <th>commision</th>
      <th>gender</th>
      <th>age</th>
      <th>claim_num</th>
      <th>F</th>
      <th>M</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CBH</td>
      <td>Travel Agency</td>
      <td>Offline</td>
      <td>Comprehensive Plan</td>
      <td>No</td>
      <td>186</td>
      <td>MALAYSIA</td>
      <td>-29.0</td>
      <td>9.57</td>
      <td>F</td>
      <td>81</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CBH</td>
      <td>Travel Agency</td>
      <td>Offline</td>
      <td>Comprehensive Plan</td>
      <td>No</td>
      <td>186</td>
      <td>MALAYSIA</td>
      <td>-29.0</td>
      <td>9.57</td>
      <td>F</td>
      <td>71</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CWT</td>
      <td>Travel Agency</td>
      <td>Online</td>
      <td>Rental Vehicle Excess Insurance</td>
      <td>No</td>
      <td>65</td>
      <td>AUSTRALIA</td>
      <td>-49.5</td>
      <td>29.70</td>
      <td>NaN</td>
      <td>32</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CWT</td>
      <td>Travel Agency</td>
      <td>Online</td>
      <td>Rental Vehicle Excess Insurance</td>
      <td>No</td>
      <td>60</td>
      <td>AUSTRALIA</td>
      <td>-39.6</td>
      <td>23.76</td>
      <td>NaN</td>
      <td>32</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CWT</td>
      <td>Travel Agency</td>
      <td>Online</td>
      <td>Rental Vehicle Excess Insurance</td>
      <td>No</td>
      <td>79</td>
      <td>ITALY</td>
      <td>-19.8</td>
      <td>11.88</td>
      <td>NaN</td>
      <td>41</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### Dealing with High Cardinality in Destination
There are a lot of values in the destination column (makes sense, lots of countries in the world!).

The following code will create a new column with frequency of visits to destination (i.e., Count of destination / number of rows in data set). The goal of this column is to indicate if more or less frequent destinations are associated with more or fewer claims(i.e., popularity of country in this data set). 

There are several notable limits to this if it were used in production:
- The desination frequency number would need to be fixed or re-calculated on a regular basis.
- If frequencies are re-calculated models would need to be re-estimted to account for changes in underlying data generating process. 
- A better soltion would be to bin on high / medium / low destination countries, but would require more complete data set. 

These limitations seem reasonable given the demo context. 


```python
dest_freq = df.groupby('destination').size()/len(df)
```


```python
# Map the frequency counts back to the destination
df.loc[:,'destination_freq'] = df['destination'].map(dest_freq)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>agency</th>
      <th>agency_type</th>
      <th>distribution_channel</th>
      <th>product_name</th>
      <th>claim</th>
      <th>duration</th>
      <th>destination</th>
      <th>net_sales</th>
      <th>commision</th>
      <th>gender</th>
      <th>age</th>
      <th>claim_num</th>
      <th>F</th>
      <th>M</th>
      <th>destination_freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CBH</td>
      <td>Travel Agency</td>
      <td>Offline</td>
      <td>Comprehensive Plan</td>
      <td>No</td>
      <td>186</td>
      <td>MALAYSIA</td>
      <td>-29.0</td>
      <td>9.57</td>
      <td>F</td>
      <td>81</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.093642</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CBH</td>
      <td>Travel Agency</td>
      <td>Offline</td>
      <td>Comprehensive Plan</td>
      <td>No</td>
      <td>186</td>
      <td>MALAYSIA</td>
      <td>-29.0</td>
      <td>9.57</td>
      <td>F</td>
      <td>71</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.093642</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CWT</td>
      <td>Travel Agency</td>
      <td>Online</td>
      <td>Rental Vehicle Excess Insurance</td>
      <td>No</td>
      <td>65</td>
      <td>AUSTRALIA</td>
      <td>-49.5</td>
      <td>29.70</td>
      <td>NaN</td>
      <td>32</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.058333</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CWT</td>
      <td>Travel Agency</td>
      <td>Online</td>
      <td>Rental Vehicle Excess Insurance</td>
      <td>No</td>
      <td>60</td>
      <td>AUSTRALIA</td>
      <td>-39.6</td>
      <td>23.76</td>
      <td>NaN</td>
      <td>32</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.058333</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CWT</td>
      <td>Travel Agency</td>
      <td>Online</td>
      <td>Rental Vehicle Excess Insurance</td>
      <td>No</td>
      <td>79</td>
      <td>ITALY</td>
      <td>-19.8</td>
      <td>11.88</td>
      <td>NaN</td>
      <td>41</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.007185</td>
    </tr>
  </tbody>
</table>
</div>



### Adjusting Skewness in Duration

To keep things simple, any duration values less than 1 or greater than 190 (95th percential) will be dropped. This is due to suspected issues with data.

Following this, a natural log transformation will be applied to duraiton to remove skewness.


```python
df = df.loc[(df.duration > 0) & (df.duration < 190)]
```


```python
df['log_duration'] = np.log(df.duration)
```


```python
sns.distplot(df.duration)
```




    <AxesSubplot:xlabel='duration', ylabel='Density'>




    
![png](tbd_insurance_data_prep_and_model_files/tbd_insurance_data_prep_and_model_19_1.png)
    



```python
sns.distplot(df.log_duration)
```




    <AxesSubplot:xlabel='log_duration', ylabel='Density'>




    
![png](tbd_insurance_data_prep_and_model_files/tbd_insurance_data_prep_and_model_20_1.png)
    


### Final data prep
The pandas profiling report indicated there was a degree of correlation between some categorical variables:
- agency and agency_type
- agency and distribution channel
- agency and product type
- product name and agency_type, distribution channel, and agency

Generally, the above suggests agencies likely specialize in offerings. For the purpose of modeling, only the following categorical variables will be kept:
- agency type (Airlines / Travel Agency)
- Distribution Channel (Online / Offline)

Product Name is being exluded due to lack of understanding. It is reasonable that these products should have a meaningful groupings, but we don't have apriori knowledge on this. As an exploratory step, we'll finish this analysis with an explortory model with all product names included as predictors. 

The final step of data preperation will convert categorical variables into numberic variables, as required for modeling. These will be automatically processed using 


```python
#making categorical column ready for model fitting
agency_type_dummy = pd.get_dummies(df['agency_type'])
dist_chan_type_dummy = pd.get_dummies(df['distribution_channel'])

df = pd.concat([df,agency_type_dummy,dist_chan_type_dummy], axis = 1)
```

Clean up some of the features after processing


```python
drop_cols = ['agency_type', 'distribution_channel', 'gender', 'destination', 
             'product_name', 'agency', 'claim','claim_num', 'duration']

X = df.drop(drop_cols, axis = 1)

```


```python
y = df['claim_num']
y.value_counts()
```




    0    59371
    1      716
    Name: claim_num, dtype: int64



### Add interaction terms for additional features


```python
X['male_x_age'] = X.age * X.M
X['female_x_age']  = X.age * X.F
X['dest_freq_x_log_duration'] = X.destination_freq * X.log_duration
X.reset_index(inplace = True, drop = True)
```

#### Apply min max scaling to prep model for random forest algorithim


```python
# Scale only columns that have values greater than 1
to_scale = [col for col in X.columns if X[col].max() > 1]
mms = MinMaxScaler()
to_scale
scaled = mms.fit_transform(X[to_scale])
scaled = pd.DataFrame(scaled, columns=to_scale)

#Replace original columns with scaled ones
for col in scaled:
    X[col] = scaled[col]
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=8187)
```

### Check to ensure rate of claims is similar between test and train sets



```python
print(f'''Positive class in Train = {np.round(y_train.value_counts(normalize=True)[1] * 100, 2)}%
Positive class in Test  = {np.round(y_test.value_counts(normalize=True)[1] * 100, 2)}%''')
```

    Positive class in Train = 1.18%
    Positive class in Test  = 1.22%



```python
mod_base = RandomForestClassifier(random_state = 8187)
mod_base = mod_base.fit(X_train, y_train)
preds = mod_base.predict(X_test)
```


```python
# Evaluate
print(f'Accuracy = {accuracy_score(y_test, preds):.2f}\nRecall = {recall_score(y_test, preds):.2f}\n')
cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(8, 6))
plt.title('Confusion Matrix (without Balancing cases)', size=16)
sns.heatmap(cm, annot=True, cmap='Blues');
```

    Accuracy = 0.99
    Recall = 0.01
    



    
![png](tbd_insurance_data_prep_and_model_files/tbd_insurance_data_prep_and_model_34_1.png)
    


Unsurprisingly, the model has very high accuracy (easy to guess no claim), but does an exceptionally poor job at correctly predicting when a claim will be filed (about 2% as indicated by recall). 

Let's try the model again, but this time using SMOTE to try to create additional simulated cases of filed claims. 
- For more information on SMOTE technique see a more detailed explanation at https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/


```python
smote = SMOTE(random_state=8187)
X_smoted, y_smoted = smote.fit_resample(X, y)
print(f'''Shape of X before SMOTE: {X.shape}
Shape of X after SMOTE: {X_smoted.shape}''')
print('\nBalance of positive and negative classes (%):')
y_smoted.value_counts(normalize=True) * 100
```

    Shape of X before SMOTE: (60087, 14)
    Shape of X after SMOTE: (118742, 14)
    
    Balance of positive and negative classes (%):





    1    50.0
    0    50.0
    Name: claim_num, dtype: float64




```python
X_train_sm, X_test_sm, y_train_sm, y_test_sm = train_test_split(
    X_smoted, y_smoted, test_size=0.25, random_state=8187
)

mod_smote = RandomForestClassifier(random_state=8187)
mod_smote.fit(X_train_sm, y_train_sm)
preds_sm = mod_smote.predict(X_test_sm)

print(f'Accuracy = {accuracy_score(y_test_sm, preds_sm):.2f}\nRecall = {recall_score(y_test_sm, preds_sm):.2f}\n')
cm = confusion_matrix(y_test_sm, preds_sm)
plt.figure(figsize=(8, 6))
plt.title('Confusion Matrix (with SMOTE)', size=16)
sns.heatmap(cm, annot=True, cmap='Blues');
```

    Accuracy = 0.98
    Recall = 0.98
    



    
![png](tbd_insurance_data_prep_and_model_files/tbd_insurance_data_prep_and_model_37_1.png)
    


### Now try the new model applied to the orginal test data. 
This will help to understand if SMOTE is introducing some artifacts inflating model performance, or if it is generally increasing the ability of our model to successfully identify claimed policies. 
- this isn't perfect as there is likely some data leakage, but an interesting exercise. 


```python
preds_sm_orginal_model = mod_smote.predict(X_test)
print(f'Accuracy = {accuracy_score(y_test, preds_sm_orginal_model):.2f}\nRecall = {recall_score(y_test, preds_sm_orginal_model):.2f}\n')

cm = confusion_matrix(y_test, preds_sm_orginal_model)
plt.figure(figsize=(8, 6))
plt.title('Confusion Matrix (with SMOTE)', size=16)
sns.heatmap(cm, annot=True, cmap='Blues');
```

    Accuracy = 0.99
    Recall = 0.80
    



    
![png](tbd_insurance_data_prep_and_model_files/tbd_insurance_data_prep_and_model_39_1.png)
    


Note the overwhelming number of predictions are that a claim is not filed, which is expected as most.  
However, there is a dramatic improvement in the model's ability to correctly identify claims actually filed. Model performance increased from predicting 2% -> 80% of filed claims correctly. 

## Conclusion
The model devloped here does a relatively good job of recovering and predicting when a policy may result in a claim being filed. The main finding demonstrated in this simple case study is that methods for dealing with class imbalance appear to be incredibly valuable to this space. Claims are relatively rare, so utilizing methods that intelligently create new synthetic cases can help to dramatically increase model performance. 

It is important to note that this is a relatively simple model, and more work would be needed before introducing into a production environment. For exmaple:
- Better understanding of destination biases in data. Singapore appeared to have a much higher incidence of claims being filed compared to other cases. 
- User level characteristics beyond basic demographics and age, may help to shed light on additional factors. 
- Understanding representativeness of agencies used in creating this data set.


```python

```
