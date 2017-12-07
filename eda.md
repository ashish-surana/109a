


```python
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score
import statsmodels.api as sm
from statsmodels.api import OLS
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import json
import pprint
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)

from IPython.display import *
```


### Loading and Cleaning with Pandas



```python
with open('dataset/review.json', 'r', encoding="utf-8") as review:
    review_data = [json.loads(line) for line in review]
    review_df = pd.DataFrame(review_data)
    print("Loaded Reviews. Total ", review_df.size, " records")

with open('dataset/business.json', 'r', encoding="utf-8") as business:
    business_data = [json.loads(line) for line in business]
    business_df = pd.DataFrame(business_data)
    print("Loaded Businesses. Total ", business_df.size, " records")
    
with open('dataset/user.json', 'r', encoding="utf-8") as user:
    user_data = [json.loads(line) for line in user]
    user_df = pd.DataFrame(user_data)
    print("Loaded Users. Total ", user_df.size, " records")
```


    Loaded Reviews. Total  42632073  records
    Loaded Businesses. Total  2349585  records
    Loaded Users. Total  26033964  records
    



```python
review_df.head(3)
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>business_id</th>
      <th>cool</th>
      <th>date</th>
      <th>funny</th>
      <th>review_id</th>
      <th>stars</th>
      <th>text</th>
      <th>useful</th>
      <th>user_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>uYHaNptLzDLoV_JZ_MuzUA</td>
      <td>0</td>
      <td>2016-07-12</td>
      <td>0</td>
      <td>VfBHSwC5Vz_pbFluy07i9Q</td>
      <td>5</td>
      <td>My girlfriend and I stayed here for 3 nights a...</td>
      <td>0</td>
      <td>cjpdDjZyprfyDG3RlkVG3w</td>
    </tr>
    <tr>
      <th>1</th>
      <td>uYHaNptLzDLoV_JZ_MuzUA</td>
      <td>0</td>
      <td>2016-10-02</td>
      <td>0</td>
      <td>3zRpneRKDsOPq92tq7ybAA</td>
      <td>3</td>
      <td>If you need an inexpensive place to stay for a...</td>
      <td>0</td>
      <td>bjTcT8Ty4cJZhEOEo01FGA</td>
    </tr>
    <tr>
      <th>2</th>
      <td>uYHaNptLzDLoV_JZ_MuzUA</td>
      <td>0</td>
      <td>2015-09-17</td>
      <td>0</td>
      <td>ne5WhI1jUFOcRn-b-gAzHA</td>
      <td>3</td>
      <td>Mittlerweile gibt es in Edinburgh zwei Ableger...</td>
      <td>0</td>
      <td>AXgRULmWcME7J6Ix3I--ww</td>
    </tr>
  </tbody>
</table>
</div>





```python
business_df.head(3)
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>address</th>
      <th>attributes</th>
      <th>business_id</th>
      <th>categories</th>
      <th>city</th>
      <th>hours</th>
      <th>is_open</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>name</th>
      <th>neighborhood</th>
      <th>postal_code</th>
      <th>review_count</th>
      <th>stars</th>
      <th>state</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>691 Richmond Rd</td>
      <td>{'RestaurantsPriceRange2': 2, 'BusinessParking...</td>
      <td>YDf95gJZaq05wvo7hTQbbQ</td>
      <td>[Shopping, Shopping Centers]</td>
      <td>Richmond Heights</td>
      <td>{'Monday': '10:00-21:00', 'Tuesday': '10:00-21...</td>
      <td>1</td>
      <td>41.541716</td>
      <td>-81.493116</td>
      <td>Richmond Town Square</td>
      <td></td>
      <td>44143</td>
      <td>17</td>
      <td>2.0</td>
      <td>OH</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2824 Milton Rd</td>
      <td>{'GoodForMeal': {'dessert': False, 'latenight'...</td>
      <td>mLwM-h2YhXl2NCgdS84_Bw</td>
      <td>[Food, Soul Food, Convenience Stores, Restaura...</td>
      <td>Charlotte</td>
      <td>{'Monday': '10:00-22:00', 'Tuesday': '10:00-22...</td>
      <td>0</td>
      <td>35.236870</td>
      <td>-80.741976</td>
      <td>South Florida Style Chicken &amp; Ribs</td>
      <td>Eastland</td>
      <td>28215</td>
      <td>4</td>
      <td>4.5</td>
      <td>NC</td>
    </tr>
    <tr>
      <th>2</th>
      <td>337 Danforth Avenue</td>
      <td>{'BusinessParking': {'garage': False, 'street'...</td>
      <td>v2WhjAB3PIBA8J8VxG3wEg</td>
      <td>[Food, Coffee &amp; Tea]</td>
      <td>Toronto</td>
      <td>{'Monday': '10:00-19:00', 'Tuesday': '10:00-19...</td>
      <td>0</td>
      <td>43.677126</td>
      <td>-79.353285</td>
      <td>The Tea Emporium</td>
      <td>Riverdale</td>
      <td>M4K 1N7</td>
      <td>7</td>
      <td>4.5</td>
      <td>ON</td>
    </tr>
  </tbody>
</table>
</div>





```python
user_df.head(3)
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>average_stars</th>
      <th>compliment_cool</th>
      <th>compliment_cute</th>
      <th>compliment_funny</th>
      <th>compliment_hot</th>
      <th>compliment_list</th>
      <th>compliment_more</th>
      <th>compliment_note</th>
      <th>compliment_photos</th>
      <th>compliment_plain</th>
      <th>compliment_profile</th>
      <th>compliment_writer</th>
      <th>cool</th>
      <th>elite</th>
      <th>fans</th>
      <th>friends</th>
      <th>funny</th>
      <th>name</th>
      <th>review_count</th>
      <th>useful</th>
      <th>user_id</th>
      <th>yelping_since</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.80</td>
      <td>5174</td>
      <td>284</td>
      <td>5174</td>
      <td>5175</td>
      <td>78</td>
      <td>299</td>
      <td>1435</td>
      <td>7829</td>
      <td>7397</td>
      <td>569</td>
      <td>1834</td>
      <td>16856</td>
      <td>[2014, 2016, 2013, 2011, 2012, 2015, 2010, 2017]</td>
      <td>209</td>
      <td>[M19NwFwAXKRZzt8koF11hQ, QRcMZ8pJJBBZaKubHOoMD...</td>
      <td>16605</td>
      <td>Cin</td>
      <td>272</td>
      <td>17019</td>
      <td>lsSiIjAKVl-QRxKjRErBeg</td>
      <td>2010-07-13</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3.94</td>
      <td>1556</td>
      <td>211</td>
      <td>1556</td>
      <td>1285</td>
      <td>101</td>
      <td>134</td>
      <td>1295</td>
      <td>162</td>
      <td>2134</td>
      <td>74</td>
      <td>402</td>
      <td>40110</td>
      <td>[2014, 2017, 2011, 2012, 2015, 2009, 2013, 200...</td>
      <td>835</td>
      <td>[eoSSJzdprj3jxXyi94vDXg, QF0urZa-0bxga17ZeY-9l...</td>
      <td>10882</td>
      <td>Andrea</td>
      <td>2559</td>
      <td>83681</td>
      <td>om5ZiponkpRqUNa3pVPiRg</td>
      <td>2006-01-18</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.72</td>
      <td>15</td>
      <td>1</td>
      <td>15</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>11</td>
      <td>8</td>
      <td>20</td>
      <td>0</td>
      <td>1</td>
      <td>55</td>
      <td>[]</td>
      <td>17</td>
      <td>[Oa84FFGBw1axX8O6uDkmqg, SRcWERSl4rhm-Bz9zN_J8...</td>
      <td>4</td>
      <td>Gabe</td>
      <td>277</td>
      <td>45</td>
      <td>-lGwMGHMC_XihFJNKCJNRg</td>
      <td>2014-10-31</td>
    </tr>
  </tbody>
</table>
</div>





```python
print(business_df.shape)
print(business_df.dtypes)
```


    (156639, 15)
    address          object
    attributes       object
    business_id      object
    categories       object
    city             object
    hours            object
    is_open           int64
    latitude        float64
    longitude       float64
    name             object
    neighborhood     object
    postal_code      object
    review_count      int64
    stars           float64
    state            object
    dtype: object
    



```python
print(user_df.shape)
print(user_df.dtypes)
```


    (1183362, 22)
    average_stars         float64
    compliment_cool         int64
    compliment_cute         int64
    compliment_funny        int64
    compliment_hot          int64
    compliment_list         int64
    compliment_more         int64
    compliment_note         int64
    compliment_photos       int64
    compliment_plain        int64
    compliment_profile      int64
    compliment_writer       int64
    cool                    int64
    elite                  object
    fans                    int64
    friends                object
    funny                   int64
    name                   object
    review_count            int64
    useful                  int64
    user_id                object
    yelping_since          object
    dtype: object
    



```python
print(review_df.shape)
print(review_df.dtypes)
```


    (4736897, 9)
    business_id    object
    cool            int64
    date           object
    funny           int64
    review_id      object
    stars           int64
    text           object
    useful          int64
    user_id        object
    dtype: object
    



```python
fig = plt.figure(figsize = (15,5))
fig.clf()
fig.subplots_adjust(hspace=.3)
ax0 = fig.add_subplot(1, 3, 1)
ax1 = fig.add_subplot(1, 3, 2) 
ax2 = fig.add_subplot(1, 3, 3) 

business_df.groupby('stars').size().plot(kind='bar', ax = ax0)
ax0.set_title('Businesses Stars Count')
ax0.set_ylabel('Number of Businesses')

business_review_count = business_df.groupby('review_count').size()
bins=[0, 5, 10, 20, 40, 80,160, 320, 640, 1280, 2560]
ax1.hist(business_review_count, bins=bins, edgecolor="k")
ax1.set_title('Businesses Reviews Count')
ax1.set_ylabel('Number of Reviews Got')
ax1.set_xscale('log')

business_review_count = user_df.groupby('review_count').size()
bins=[0, 5, 10, 20, 40, 80,160, 320, 640, 1280, 2560]
ax2.hist(business_review_count, bins=bins, edgecolor="k")
ax2.set_title('User Reviews Count')
ax2.set_ylabel('Number of Reviews Give')
ax2.set_xscale('log')

fig.savefig('plot1.png')
```



![png](eda_files/eda_9_0.png)




```python
fig = plt.figure(figsize = (18,5))
fig.clf()
fig.subplots_adjust(hspace=.3)
ax0 = fig.add_subplot(1, 3, 1)
ax1 = fig.add_subplot(1, 3, 2) 
ax2 = fig.add_subplot(1, 3, 3) 

ax0.scatter(business_df['review_count'], business_df['stars'])
ax0.set_title('review_count - stars relation')
ax0.set_xlabel('review_count')
ax0.set_ylabel('stars')

business_df[['state', 'stars']].groupby('state')['stars'].agg('mean').sort_values(ascending = False).head(15).plot(kind = 'bar', ax=ax1)
ax1.set_title('Top states with star ratings')
ax1.set_ylabel('Stars')

business_df[['state', 'stars']].groupby('state')['stars'].count().sort_values(ascending = False).head(15).plot(kind = 'bar', ax=ax2)
ax2.set_title('Top states with highest number of reviews')
ax2.set_ylabel('Number of reviews')

fig.savefig('plot2.png')
```



![png](eda_files/eda_10_0.png)


### Data Selection/Cleaning



```python
# First of all let's filter out closed businesses 
open_business_df = business_df[business_df['is_open'] == 1]
print("After removing businesses that are closed we left with ", open_business_df.size, " records")

# Next, filter out all none restaurant businesses, because we only care about restaurants
restaurant_df = open_business_df[open_business_df['categories'].apply(lambda x: 'Restaurants' in x)]
print("Open restaurant business records: ", restaurant_df.size)
```


    After removing businesses that are closed we left with  1983930  records
    Open restaurant business records:  579855
    



```python
restaurant_df['business_hours'][:1]
```


    14    {'Monday': '11:00-0:00', 'Tuesday': '11:00-0:0...
    Name: business_hours, dtype: object
    



```python
print(len(restaurant_hours))
print(len(restaurant_df['hours']) == len(restaurant_hours))

# Work in progress on adding hour columns before training
```


    38657
    True
    



```python
print(restaurant_df.dtypes)
```


    business_address          object
    business_attributes       object
    business_id               object
    business_categories       object
    business_city             object
    business_hours            object
    business_is_open           int64
    business_latitude        float64
    business_longitude       float64
    business_name             object
    business_neighborhood     object
    business_postal_code      object
    business_review_count      int64
    business_stars           float64
    business_state            object
    dtype: object
    



```python
# Append `business_` and `review_` prefix to all columns in restaurants and reviews dataframe 
# to distinguish columns after merge
restaurant_df.columns = ['business_' + str(col) for col in restaurant_df.columns]
review_df.columns = ['review_' + str(col) for col in review_df.columns]
# rename *_id columns back 
restaurant_df.rename(columns={"business_business_id": "business_id"}, inplace=True)
review_df.rename(columns={"review_business_id": "business_id", 
                          "review_review_id": "review_id", 
                          "review_user_id": "user_id"}, inplace=True)
```


    /Users/rburdakov/anaconda/envs/3point6/lib/python3.6/site-packages/pandas/core/frame.py:2746: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      **kwargs)
    



```python
yelp_reviews = pd.merge(pd.merge(restaurant_df, review_df, on='business_id', how='left'),
              user_df, on='user_id', how='left')
yelp_reviews.head(3)
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>business_address</th>
      <th>business_attributes</th>
      <th>business_id</th>
      <th>business_categories</th>
      <th>business_city</th>
      <th>business_hours</th>
      <th>business_is_open</th>
      <th>business_latitude</th>
      <th>business_longitude</th>
      <th>business_name</th>
      <th>business_neighborhood</th>
      <th>business_postal_code</th>
      <th>business_review_count</th>
      <th>business_stars</th>
      <th>business_state</th>
      <th>review_cool</th>
      <th>review_date</th>
      <th>review_funny</th>
      <th>review_id</th>
      <th>review_stars</th>
      <th>review_text</th>
      <th>review_useful</th>
      <th>user_id</th>
      <th>average_stars</th>
      <th>compliment_cool</th>
      <th>compliment_cute</th>
      <th>compliment_funny</th>
      <th>compliment_hot</th>
      <th>compliment_list</th>
      <th>compliment_more</th>
      <th>compliment_note</th>
      <th>compliment_photos</th>
      <th>compliment_plain</th>
      <th>compliment_profile</th>
      <th>compliment_writer</th>
      <th>cool</th>
      <th>elite</th>
      <th>fans</th>
      <th>friends</th>
      <th>funny</th>
      <th>name</th>
      <th>review_count</th>
      <th>useful</th>
      <th>yelping_since</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9616 E Independence Blvd</td>
      <td>{'Alcohol': 'full_bar', 'HasTV': True, 'NoiseL...</td>
      <td>SDMRxmcKPNt1AHPBKqO64Q</td>
      <td>[Burgers, Bars, Restaurants, Sports Bars, Nigh...</td>
      <td>Matthews</td>
      <td>{'Monday': '11:00-0:00', 'Tuesday': '11:00-0:0...</td>
      <td>1</td>
      <td>35.135196</td>
      <td>-80.714683</td>
      <td>Applebee's</td>
      <td></td>
      <td>28105</td>
      <td>21</td>
      <td>2.0</td>
      <td>NC</td>
      <td>0</td>
      <td>2016-04-06</td>
      <td>0</td>
      <td>EBTHgI_19gtQfivTJlsPkA</td>
      <td>2</td>
      <td>I hadn't been to a Applebee's for a few years....</td>
      <td>0</td>
      <td>M0cI78odeq_GKqLzk8sIrw</td>
      <td>3.49</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>[]</td>
      <td>0</td>
      <td>[EPDSZUPujQGhluDYdf55hw, BTGB7ZoCMCclhZL0gd9SI...</td>
      <td>1</td>
      <td>Murray</td>
      <td>49</td>
      <td>10</td>
      <td>2010-11-09</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9616 E Independence Blvd</td>
      <td>{'Alcohol': 'full_bar', 'HasTV': True, 'NoiseL...</td>
      <td>SDMRxmcKPNt1AHPBKqO64Q</td>
      <td>[Burgers, Bars, Restaurants, Sports Bars, Nigh...</td>
      <td>Matthews</td>
      <td>{'Monday': '11:00-0:00', 'Tuesday': '11:00-0:0...</td>
      <td>1</td>
      <td>35.135196</td>
      <td>-80.714683</td>
      <td>Applebee's</td>
      <td></td>
      <td>28105</td>
      <td>21</td>
      <td>2.0</td>
      <td>NC</td>
      <td>0</td>
      <td>2016-04-10</td>
      <td>0</td>
      <td>fT9506-dhjrMTmYLJO0xmg</td>
      <td>3</td>
      <td>I am an avid Applebees fan. In high school I u...</td>
      <td>0</td>
      <td>4i0NQ2eyuQZKpXbz8TxBEg</td>
      <td>4.10</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>[]</td>
      <td>0</td>
      <td>[d9Ls7HeX4dBqlubUBf1PoQ, nOcoxf8AAYpT4hbWiziVp...</td>
      <td>1</td>
      <td>Katie</td>
      <td>20</td>
      <td>0</td>
      <td>2010-01-01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9616 E Independence Blvd</td>
      <td>{'Alcohol': 'full_bar', 'HasTV': True, 'NoiseL...</td>
      <td>SDMRxmcKPNt1AHPBKqO64Q</td>
      <td>[Burgers, Bars, Restaurants, Sports Bars, Nigh...</td>
      <td>Matthews</td>
      <td>{'Monday': '11:00-0:00', 'Tuesday': '11:00-0:0...</td>
      <td>1</td>
      <td>35.135196</td>
      <td>-80.714683</td>
      <td>Applebee's</td>
      <td></td>
      <td>28105</td>
      <td>21</td>
      <td>2.0</td>
      <td>NC</td>
      <td>0</td>
      <td>2017-03-22</td>
      <td>0</td>
      <td>-Ojqi_nKPwl8HhN8ShzlaA</td>
      <td>4</td>
      <td>Pleasantly surprised.   Better than expected a...</td>
      <td>0</td>
      <td>kABsypSKvgLPkqd2YIWB8Q</td>
      <td>4.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>[]</td>
      <td>0</td>
      <td>[NMcdjEAbH1G1SSDu8T4AGA]</td>
      <td>0</td>
      <td>Howard</td>
      <td>1</td>
      <td>0</td>
      <td>2014-10-03</td>
    </tr>
  </tbody>
</table>
</div>





```python
# save_msk = np.random.rand(len(yelp_reviews)) < 0.6
# yelp_reviews[save_msk].to_csv('dataset/restaurant_yelp_reviews.csv')
```




```python
print("Merged data frame contains ", yelp_reviews.columns.size, " columns:\n\n", yelp_reviews.columns.tolist())
```


    Merged data frame contains  44  columns:
    
     ['business_address', 'business_attributes', 'business_id', 'business_categories', 'business_city', 'business_hours', 'business_is_open', 'business_latitude', 'business_longitude', 'business_name', 'business_neighborhood', 'business_postal_code', 'business_review_count', 'business_stars', 'business_state', 'review_cool', 'review_date', 'review_funny', 'review_id', 'review_stars', 'review_text', 'review_useful', 'user_id', 'average_stars', 'compliment_cool', 'compliment_cute', 'compliment_funny', 'compliment_hot', 'compliment_list', 'compliment_more', 'compliment_note', 'compliment_photos', 'compliment_plain', 'compliment_profile', 'compliment_writer', 'cool', 'elite', 'fans', 'friends', 'funny', 'name', 'review_count', 'useful', 'yelping_since']
    



```python
# We need to process hours column to factor out time
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
open_close = ['_open', '_close']

def get_hours(df):
    hours = []
    for index, row in df.iterrows():
        record = {'business_id': row['business_id']}
        s = row['business_hours']
        for d in days:
            opn = 0.0
            cls = 0.0
            if d in s:
                hourz = s[d].split('-')
                hrs1 = hourz[0].split(':')
                hrs2 = hourz[1].split(':')

                opn = float(hrs1[0]) + float(hrs1[1])/60
                cls = float(hrs2[0]) + float(hrs2[1])/60

                # handle overnight hours
                if (opn > cls):
                    cls += 24

            record[str(d) + '_open'] = opn
            record[str(d) + '_close'] = cls
            
        hours.append(record)
    return hours

restaurant_hours = get_hours(restaurant_df[['business_id', 'business_hours']])
```




```python
restaurant_hours_df = pd.DataFrame(restaurant_hours)
restaurant_df_merged = restaurant_df.merge(restaurant_hours_df, on=['business_id'])
restaurant_df_merged.head(3)
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>business_address</th>
      <th>business_attributes</th>
      <th>business_id</th>
      <th>business_categories</th>
      <th>business_city</th>
      <th>business_hours</th>
      <th>business_is_open</th>
      <th>business_latitude</th>
      <th>business_longitude</th>
      <th>business_name</th>
      <th>business_neighborhood</th>
      <th>business_postal_code</th>
      <th>business_review_count</th>
      <th>business_stars</th>
      <th>business_state</th>
      <th>Friday_close</th>
      <th>Friday_open</th>
      <th>Monday_close</th>
      <th>Monday_open</th>
      <th>Saturday_close</th>
      <th>Saturday_open</th>
      <th>Sunday_close</th>
      <th>Sunday_open</th>
      <th>Thursday_close</th>
      <th>Thursday_open</th>
      <th>Tuesday_close</th>
      <th>Tuesday_open</th>
      <th>Wednesday_close</th>
      <th>Wednesday_open</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9616 E Independence Blvd</td>
      <td>{'Alcohol': 'full_bar', 'HasTV': True, 'NoiseL...</td>
      <td>SDMRxmcKPNt1AHPBKqO64Q</td>
      <td>[Burgers, Bars, Restaurants, Sports Bars, Nigh...</td>
      <td>Matthews</td>
      <td>{'Monday': '11:00-0:00', 'Tuesday': '11:00-0:0...</td>
      <td>1</td>
      <td>35.135196</td>
      <td>-80.714683</td>
      <td>Applebee's</td>
      <td></td>
      <td>28105</td>
      <td>21</td>
      <td>2.0</td>
      <td>NC</td>
      <td>25.0</td>
      <td>11.0</td>
      <td>24.0</td>
      <td>11.0</td>
      <td>25.0</td>
      <td>11.0</td>
      <td>24.0</td>
      <td>11.0</td>
      <td>24.0</td>
      <td>11.0</td>
      <td>24.0</td>
      <td>11.0</td>
      <td>24.0</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>190 E Dallas Rd</td>
      <td>{'RestaurantsAttire': 'casual', 'Alcohol': 'no...</td>
      <td>iFEiMJoEqyB9O8OUNSdLzA</td>
      <td>[Chinese, Restaurants]</td>
      <td>Stanley</td>
      <td>{'Monday': '11:30-22:00', 'Tuesday': '11:30-22...</td>
      <td>1</td>
      <td>35.355085</td>
      <td>-81.087268</td>
      <td>China Garden</td>
      <td></td>
      <td>28164</td>
      <td>3</td>
      <td>3.0</td>
      <td>NC</td>
      <td>22.5</td>
      <td>11.5</td>
      <td>22.0</td>
      <td>11.5</td>
      <td>22.5</td>
      <td>11.5</td>
      <td>22.0</td>
      <td>11.5</td>
      <td>22.0</td>
      <td>11.5</td>
      <td>22.0</td>
      <td>11.5</td>
      <td>22.0</td>
      <td>11.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4759 Liberty Ave</td>
      <td>{'RestaurantsTableService': True, 'GoodForMeal...</td>
      <td>HmI9nhgOkrXlUr6KZGZZew</td>
      <td>[Sandwiches, Restaurants, Italian, Diners, Bre...</td>
      <td>Pittsburgh</td>
      <td>{'Sunday': '8:00-12:00', 'Tuesday': '8:00-12:0...</td>
      <td>1</td>
      <td>40.461350</td>
      <td>-79.948113</td>
      <td>Rocky's</td>
      <td>Bloomfield</td>
      <td>15224</td>
      <td>15</td>
      <td>3.0</td>
      <td>PA</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>12.0</td>
      <td>8.0</td>
      <td>12.0</td>
      <td>8.0</td>
      <td>12.0</td>
      <td>8.0</td>
      <td>12.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>





```python
# Next let's take a look at all categories that has 'Restaurant'
categories = set()
restaurant_df_merged['business_categories'].apply(lambda r: categories.update(r))
len(categories)
```





    635





```python
def process_categories(df):
    records = []
    for index, row in df.iterrows():
        record = {'business_id': row['business_id']}
        current_cats = row['business_categories']
        for c in current_cats:
            record[c] = 1
        records.append(record)
    return records

b_cats = process_categories(restaurant_df_merged)
```




```python
cats_df = pd.DataFrame(b_cats).fillna(0)
cats_df.head(3)
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Acai Bowls</th>
      <th>Accessories</th>
      <th>Accountants</th>
      <th>Active Life</th>
      <th>Acupuncture</th>
      <th>Adult</th>
      <th>Adult Education</th>
      <th>Adult Entertainment</th>
      <th>Advertising</th>
      <th>Afghan</th>
      <th>African</th>
      <th>Air Duct Cleaning</th>
      <th>Aircraft Repairs</th>
      <th>Airport Lounges</th>
      <th>Airport Shuttles</th>
      <th>Airports</th>
      <th>Airsoft</th>
      <th>Alsatian</th>
      <th>Amateur Sports Teams</th>
      <th>American (New)</th>
      <th>American (Traditional)</th>
      <th>Amusement Parks</th>
      <th>Animal Shelters</th>
      <th>Antiques</th>
      <th>Apartments</th>
      <th>Appliances</th>
      <th>Appliances &amp; Repair</th>
      <th>Aquarium Services</th>
      <th>Aquariums</th>
      <th>Arabian</th>
      <th>Arcades</th>
      <th>Argentine</th>
      <th>Armenian</th>
      <th>Art Classes</th>
      <th>Art Galleries</th>
      <th>Art Schools</th>
      <th>Arts &amp; Crafts</th>
      <th>Arts &amp; Entertainment</th>
      <th>Asian Fusion</th>
      <th>Australian</th>
      <th>Austrian</th>
      <th>Auto Customization</th>
      <th>Auto Detailing</th>
      <th>Auto Glass Services</th>
      <th>Auto Insurance</th>
      <th>Auto Parts &amp; Supplies</th>
      <th>Auto Repair</th>
      <th>Auto Upholstery</th>
      <th>Automotive</th>
      <th>Baby Gear &amp; Furniture</th>
      <th>...</th>
      <th>Trinidadian</th>
      <th>Truck Rental</th>
      <th>Turkish</th>
      <th>Tuscan</th>
      <th>Udon</th>
      <th>Ukrainian</th>
      <th>Used</th>
      <th>Uzbek</th>
      <th>Vacation Rentals</th>
      <th>Vape Shops</th>
      <th>Vegan</th>
      <th>Vegetarian</th>
      <th>Vehicle Wraps</th>
      <th>Venezuelan</th>
      <th>Venues &amp; Event Spaces</th>
      <th>Veterinarians</th>
      <th>Video Game Stores</th>
      <th>Videos &amp; Video Game Rental</th>
      <th>Vietnamese</th>
      <th>Vintage &amp; Consignment</th>
      <th>Vinyl Records</th>
      <th>Vinyl Siding</th>
      <th>Vitamins &amp; Supplements</th>
      <th>Waffles</th>
      <th>Walking Tours</th>
      <th>Water Heater Installation/Repair</th>
      <th>Water Stores</th>
      <th>Waxing</th>
      <th>Web Design</th>
      <th>Wedding Chapels</th>
      <th>Wedding Planning</th>
      <th>Weight Loss Centers</th>
      <th>Whiskey Bars</th>
      <th>Wholesale Stores</th>
      <th>Wholesalers</th>
      <th>Wigs</th>
      <th>Windows Installation</th>
      <th>Windshield Installation &amp; Repair</th>
      <th>Wine &amp; Spirits</th>
      <th>Wine Bars</th>
      <th>Wine Tasting Room</th>
      <th>Wine Tours</th>
      <th>Wineries</th>
      <th>Wok</th>
      <th>Women's Clothing</th>
      <th>Wraps</th>
      <th>Yelp Events</th>
      <th>Yoga</th>
      <th>Zoos</th>
      <th>business_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>SDMRxmcKPNt1AHPBKqO64Q</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>iFEiMJoEqyB9O8OUNSdLzA</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>HmI9nhgOkrXlUr6KZGZZew</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 636 columns</p>
</div>





```python
restaurant_df_merged = restaurant_df_merged.merge(cats_df, on=['business_id'])
restaurant_df_merged.head(3)
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>business_address</th>
      <th>business_attributes</th>
      <th>business_id</th>
      <th>business_categories</th>
      <th>business_city</th>
      <th>business_hours</th>
      <th>business_is_open</th>
      <th>business_latitude</th>
      <th>business_longitude</th>
      <th>business_name</th>
      <th>business_neighborhood</th>
      <th>business_postal_code</th>
      <th>business_review_count</th>
      <th>business_stars</th>
      <th>business_state</th>
      <th>Friday_close</th>
      <th>Friday_open</th>
      <th>Monday_close</th>
      <th>Monday_open</th>
      <th>Saturday_close</th>
      <th>Saturday_open</th>
      <th>Sunday_close</th>
      <th>Sunday_open</th>
      <th>Thursday_close</th>
      <th>Thursday_open</th>
      <th>Tuesday_close</th>
      <th>Tuesday_open</th>
      <th>Wednesday_close</th>
      <th>Wednesday_open</th>
      <th>Acai Bowls</th>
      <th>Accessories</th>
      <th>Accountants</th>
      <th>Active Life</th>
      <th>Acupuncture</th>
      <th>Adult</th>
      <th>Adult Education</th>
      <th>Adult Entertainment</th>
      <th>Advertising</th>
      <th>Afghan</th>
      <th>African</th>
      <th>Air Duct Cleaning</th>
      <th>Aircraft Repairs</th>
      <th>Airport Lounges</th>
      <th>Airport Shuttles</th>
      <th>Airports</th>
      <th>Airsoft</th>
      <th>Alsatian</th>
      <th>Amateur Sports Teams</th>
      <th>American (New)</th>
      <th>American (Traditional)</th>
      <th>...</th>
      <th>Travel Services</th>
      <th>Trinidadian</th>
      <th>Truck Rental</th>
      <th>Turkish</th>
      <th>Tuscan</th>
      <th>Udon</th>
      <th>Ukrainian</th>
      <th>Used</th>
      <th>Uzbek</th>
      <th>Vacation Rentals</th>
      <th>Vape Shops</th>
      <th>Vegan</th>
      <th>Vegetarian</th>
      <th>Vehicle Wraps</th>
      <th>Venezuelan</th>
      <th>Venues &amp; Event Spaces</th>
      <th>Veterinarians</th>
      <th>Video Game Stores</th>
      <th>Videos &amp; Video Game Rental</th>
      <th>Vietnamese</th>
      <th>Vintage &amp; Consignment</th>
      <th>Vinyl Records</th>
      <th>Vinyl Siding</th>
      <th>Vitamins &amp; Supplements</th>
      <th>Waffles</th>
      <th>Walking Tours</th>
      <th>Water Heater Installation/Repair</th>
      <th>Water Stores</th>
      <th>Waxing</th>
      <th>Web Design</th>
      <th>Wedding Chapels</th>
      <th>Wedding Planning</th>
      <th>Weight Loss Centers</th>
      <th>Whiskey Bars</th>
      <th>Wholesale Stores</th>
      <th>Wholesalers</th>
      <th>Wigs</th>
      <th>Windows Installation</th>
      <th>Windshield Installation &amp; Repair</th>
      <th>Wine &amp; Spirits</th>
      <th>Wine Bars</th>
      <th>Wine Tasting Room</th>
      <th>Wine Tours</th>
      <th>Wineries</th>
      <th>Wok</th>
      <th>Women's Clothing</th>
      <th>Wraps</th>
      <th>Yelp Events</th>
      <th>Yoga</th>
      <th>Zoos</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9616 E Independence Blvd</td>
      <td>{'Alcohol': 'full_bar', 'HasTV': True, 'NoiseL...</td>
      <td>SDMRxmcKPNt1AHPBKqO64Q</td>
      <td>[Burgers, Bars, Restaurants, Sports Bars, Nigh...</td>
      <td>Matthews</td>
      <td>{'Monday': '11:00-0:00', 'Tuesday': '11:00-0:0...</td>
      <td>1</td>
      <td>35.135196</td>
      <td>-80.714683</td>
      <td>Applebee's</td>
      <td></td>
      <td>28105</td>
      <td>21</td>
      <td>2.0</td>
      <td>NC</td>
      <td>25.0</td>
      <td>11.0</td>
      <td>24.0</td>
      <td>11.0</td>
      <td>25.0</td>
      <td>11.0</td>
      <td>24.0</td>
      <td>11.0</td>
      <td>24.0</td>
      <td>11.0</td>
      <td>24.0</td>
      <td>11.0</td>
      <td>24.0</td>
      <td>11.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>190 E Dallas Rd</td>
      <td>{'RestaurantsAttire': 'casual', 'Alcohol': 'no...</td>
      <td>iFEiMJoEqyB9O8OUNSdLzA</td>
      <td>[Chinese, Restaurants]</td>
      <td>Stanley</td>
      <td>{'Monday': '11:30-22:00', 'Tuesday': '11:30-22...</td>
      <td>1</td>
      <td>35.355085</td>
      <td>-81.087268</td>
      <td>China Garden</td>
      <td></td>
      <td>28164</td>
      <td>3</td>
      <td>3.0</td>
      <td>NC</td>
      <td>22.5</td>
      <td>11.5</td>
      <td>22.0</td>
      <td>11.5</td>
      <td>22.5</td>
      <td>11.5</td>
      <td>22.0</td>
      <td>11.5</td>
      <td>22.0</td>
      <td>11.5</td>
      <td>22.0</td>
      <td>11.5</td>
      <td>22.0</td>
      <td>11.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4759 Liberty Ave</td>
      <td>{'RestaurantsTableService': True, 'GoodForMeal...</td>
      <td>HmI9nhgOkrXlUr6KZGZZew</td>
      <td>[Sandwiches, Restaurants, Italian, Diners, Bre...</td>
      <td>Pittsburgh</td>
      <td>{'Sunday': '8:00-12:00', 'Tuesday': '8:00-12:0...</td>
      <td>1</td>
      <td>40.461350</td>
      <td>-79.948113</td>
      <td>Rocky's</td>
      <td>Bloomfield</td>
      <td>15224</td>
      <td>15</td>
      <td>3.0</td>
      <td>PA</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>12.0</td>
      <td>8.0</td>
      <td>12.0</td>
      <td>8.0</td>
      <td>12.0</td>
      <td>8.0</td>
      <td>12.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 664 columns</p>
</div>





```python
restaurant_df_merged.isnull().values.any()
```





    False





```python
def process_attributes(df):
    records = []
    for i, row in df.iterrows():
        attrs = row['business_attributes']
        for key, val in attrs.items():
            if key == 'Alcohol':
                if val == 'full_bar':
                    attrs[key] = True
                else:
                    attrs[key] = False
                    
            if key == 'Smoking':
                # we will treat outdoor as not allowed 
                if val == 'no' or val == 'outdoor':
                    attrs[key] = False
                else:
                    attrs[key] = True
        
        for k, v in attrs.get('BusinessParking', {}).items():
            attrs['BusinessParking_' + k] = v
           
        for k, v in attrs.get('Ambience', {}).items():
            attrs['Ambience_' + k] = v
            
        for k, v in attrs.get('GoodForMeal', {}).items():
            attrs['GoodForMeal_' + k] = v
        
        for k, v in attrs.get('Music', {}).items():
            attrs['Music_' + k] = v
        
        # removed records that we have processed as well as remove 
        # HairSpecializesIn which isn't relevant for restaurants
        for k in ['BusinessParking', 'Ambience', 'GoodForMeal', 
                  'Music', 'HairSpecializesIn']: 
            attrs.pop(k, None)
                
        attrs['business_id'] = row['business_id']
        records.append(attrs)
    return records

b_attrs = process_attributes(restaurant_df_merged)
```




```python

```




```python
pd.DataFrame(b_attrs)['ByAppointmentOnly'].unique()
```





    array([nan, False, True], dtype=object)





```python
pd.DataFrame(b_attrs).head(5)
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AcceptsInsurance</th>
      <th>AgesAllowed</th>
      <th>Alcohol</th>
      <th>Ambience_casual</th>
      <th>Ambience_classy</th>
      <th>Ambience_divey</th>
      <th>Ambience_hipster</th>
      <th>Ambience_intimate</th>
      <th>Ambience_romantic</th>
      <th>Ambience_touristy</th>
      <th>Ambience_trendy</th>
      <th>Ambience_upscale</th>
      <th>BYOB</th>
      <th>BYOBCorkage</th>
      <th>BestNights</th>
      <th>BikeParking</th>
      <th>BusinessAcceptsBitcoin</th>
      <th>BusinessAcceptsCreditCards</th>
      <th>BusinessParking_garage</th>
      <th>BusinessParking_lot</th>
      <th>BusinessParking_street</th>
      <th>BusinessParking_valet</th>
      <th>BusinessParking_validated</th>
      <th>ByAppointmentOnly</th>
      <th>Caters</th>
      <th>CoatCheck</th>
      <th>Corkage</th>
      <th>DietaryRestrictions</th>
      <th>DogsAllowed</th>
      <th>DriveThru</th>
      <th>GoodForDancing</th>
      <th>GoodForKids</th>
      <th>GoodForMeal_breakfast</th>
      <th>GoodForMeal_brunch</th>
      <th>GoodForMeal_dessert</th>
      <th>GoodForMeal_dinner</th>
      <th>GoodForMeal_latenight</th>
      <th>GoodForMeal_lunch</th>
      <th>HappyHour</th>
      <th>HasTV</th>
      <th>Music_background_music</th>
      <th>Music_dj</th>
      <th>Music_jukebox</th>
      <th>Music_karaoke</th>
      <th>Music_live</th>
      <th>Music_no_music</th>
      <th>Music_video</th>
      <th>NoiseLevel</th>
      <th>Open24Hours</th>
      <th>OutdoorSeating</th>
      <th>RestaurantsAttire</th>
      <th>RestaurantsCounterService</th>
      <th>RestaurantsDelivery</th>
      <th>RestaurantsGoodForGroups</th>
      <th>RestaurantsPriceRange2</th>
      <th>RestaurantsReservations</th>
      <th>RestaurantsTableService</th>
      <th>RestaurantsTakeOut</th>
      <th>Smoking</th>
      <th>WheelchairAccessible</th>
      <th>WiFi</th>
      <th>breakfast</th>
      <th>brunch</th>
      <th>business_id</th>
      <th>casual</th>
      <th>classy</th>
      <th>dessert</th>
      <th>dinner</th>
      <th>divey</th>
      <th>garage</th>
      <th>hipster</th>
      <th>intimate</th>
      <th>latenight</th>
      <th>lot</th>
      <th>lunch</th>
      <th>romantic</th>
      <th>street</th>
      <th>touristy</th>
      <th>trendy</th>
      <th>upscale</th>
      <th>valet</th>
      <th>validated</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>NaN</td>
      <td>False</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>average</td>
      <td>NaN</td>
      <td>False</td>
      <td>casual</td>
      <td>NaN</td>
      <td>False</td>
      <td>True</td>
      <td>2.0</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>free</td>
      <td>False</td>
      <td>False</td>
      <td>SDMRxmcKPNt1AHPBKqO64Q</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>casual</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>iFEiMJoEqyB9O8OUNSdLzA</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>loud</td>
      <td>NaN</td>
      <td>False</td>
      <td>casual</td>
      <td>NaN</td>
      <td>False</td>
      <td>False</td>
      <td>1.0</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>no</td>
      <td>True</td>
      <td>False</td>
      <td>HmI9nhgOkrXlUr6KZGZZew</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>average</td>
      <td>NaN</td>
      <td>True</td>
      <td>casual</td>
      <td>NaN</td>
      <td>False</td>
      <td>True</td>
      <td>2.0</td>
      <td>True</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>free</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>qnpvw-uQyRn9nlClWFK9aA</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>average</td>
      <td>NaN</td>
      <td>False</td>
      <td>casual</td>
      <td>NaN</td>
      <td>False</td>
      <td>True</td>
      <td>2.0</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>NaN</td>
      <td>True</td>
      <td>no</td>
      <td>False</td>
      <td>False</td>
      <td>TXiEgINSZ75d3EtvLvkc4Q</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>





```python
# those columns are ready
# 'ByAppointmentOnly', 'Caters', 'CoatCheck', 'Corkage', 'DogsAllowed', 'DriveThru','GoodForDancing', 'GoodForKids', 'HappyHour', 'HasTV', 'OutdoorSeating', 'Open24Hours', 'RestaurantsTakeOut', 'RestaurantsTableService', 'RestaurantsReservations',
# 'RestaurantsPriceRange2', 'RestaurantsGoodForGroups', 'RestaurantsDelivery', 'RestaurantsCounterService', 
# 'WiFi', 'RestaurantsAttire', 'NoiseLevel'


# this needs to be categorized
# 'DietaryRestrictions', 'BusinessParking', 
```




```python

```




```python
restaurant_df_merged.drop(['business_attributes', 'business_hours', 'business_categories'], axis=1)
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>business_address</th>
      <th>business_id</th>
      <th>business_city</th>
      <th>business_is_open</th>
      <th>business_latitude</th>
      <th>business_longitude</th>
      <th>business_name</th>
      <th>business_neighborhood</th>
      <th>business_postal_code</th>
      <th>business_review_count</th>
      <th>business_stars</th>
      <th>business_state</th>
      <th>Friday_close</th>
      <th>Friday_open</th>
      <th>Monday_close</th>
      <th>Monday_open</th>
      <th>Saturday_close</th>
      <th>Saturday_open</th>
      <th>Sunday_close</th>
      <th>Sunday_open</th>
      <th>Thursday_close</th>
      <th>Thursday_open</th>
      <th>Tuesday_close</th>
      <th>Tuesday_open</th>
      <th>Wednesday_close</th>
      <th>Wednesday_open</th>
      <th>Acai Bowls</th>
      <th>Accessories</th>
      <th>Accountants</th>
      <th>Active Life</th>
      <th>Acupuncture</th>
      <th>Adult</th>
      <th>Adult Education</th>
      <th>Adult Entertainment</th>
      <th>Advertising</th>
      <th>Afghan</th>
      <th>African</th>
      <th>Air Duct Cleaning</th>
      <th>Aircraft Repairs</th>
      <th>Airport Lounges</th>
      <th>Airport Shuttles</th>
      <th>Airports</th>
      <th>Airsoft</th>
      <th>Alsatian</th>
      <th>Amateur Sports Teams</th>
      <th>American (New)</th>
      <th>American (Traditional)</th>
      <th>Amusement Parks</th>
      <th>Animal Shelters</th>
      <th>Antiques</th>
      <th>...</th>
      <th>Travel Services</th>
      <th>Trinidadian</th>
      <th>Truck Rental</th>
      <th>Turkish</th>
      <th>Tuscan</th>
      <th>Udon</th>
      <th>Ukrainian</th>
      <th>Used</th>
      <th>Uzbek</th>
      <th>Vacation Rentals</th>
      <th>Vape Shops</th>
      <th>Vegan</th>
      <th>Vegetarian</th>
      <th>Vehicle Wraps</th>
      <th>Venezuelan</th>
      <th>Venues &amp; Event Spaces</th>
      <th>Veterinarians</th>
      <th>Video Game Stores</th>
      <th>Videos &amp; Video Game Rental</th>
      <th>Vietnamese</th>
      <th>Vintage &amp; Consignment</th>
      <th>Vinyl Records</th>
      <th>Vinyl Siding</th>
      <th>Vitamins &amp; Supplements</th>
      <th>Waffles</th>
      <th>Walking Tours</th>
      <th>Water Heater Installation/Repair</th>
      <th>Water Stores</th>
      <th>Waxing</th>
      <th>Web Design</th>
      <th>Wedding Chapels</th>
      <th>Wedding Planning</th>
      <th>Weight Loss Centers</th>
      <th>Whiskey Bars</th>
      <th>Wholesale Stores</th>
      <th>Wholesalers</th>
      <th>Wigs</th>
      <th>Windows Installation</th>
      <th>Windshield Installation &amp; Repair</th>
      <th>Wine &amp; Spirits</th>
      <th>Wine Bars</th>
      <th>Wine Tasting Room</th>
      <th>Wine Tours</th>
      <th>Wineries</th>
      <th>Wok</th>
      <th>Women's Clothing</th>
      <th>Wraps</th>
      <th>Yelp Events</th>
      <th>Yoga</th>
      <th>Zoos</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9616 E Independence Blvd</td>
      <td>SDMRxmcKPNt1AHPBKqO64Q</td>
      <td>Matthews</td>
      <td>1</td>
      <td>35.135196</td>
      <td>-80.714683</td>
      <td>Applebee's</td>
      <td></td>
      <td>28105</td>
      <td>21</td>
      <td>2.0</td>
      <td>NC</td>
      <td>25.0</td>
      <td>11.0</td>
      <td>24.0</td>
      <td>11.0</td>
      <td>25.0</td>
      <td>11.00</td>
      <td>24.0</td>
      <td>11.00</td>
      <td>24.0</td>
      <td>11.0</td>
      <td>24.0</td>
      <td>11.0</td>
      <td>24.0</td>
      <td>11.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>190 E Dallas Rd</td>
      <td>iFEiMJoEqyB9O8OUNSdLzA</td>
      <td>Stanley</td>
      <td>1</td>
      <td>35.355085</td>
      <td>-81.087268</td>
      <td>China Garden</td>
      <td></td>
      <td>28164</td>
      <td>3</td>
      <td>3.0</td>
      <td>NC</td>
      <td>22.5</td>
      <td>11.5</td>
      <td>22.0</td>
      <td>11.5</td>
      <td>22.5</td>
      <td>11.50</td>
      <td>22.0</td>
      <td>11.50</td>
      <td>22.0</td>
      <td>11.5</td>
      <td>22.0</td>
      <td>11.5</td>
      <td>22.0</td>
      <td>11.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4759 Liberty Ave</td>
      <td>HmI9nhgOkrXlUr6KZGZZew</td>
      <td>Pittsburgh</td>
      <td>1</td>
      <td>40.461350</td>
      <td>-79.948113</td>
      <td>Rocky's</td>
      <td>Bloomfield</td>
      <td>15224</td>
      <td>15</td>
      <td>3.0</td>
      <td>PA</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>12.0</td>
      <td>8.00</td>
      <td>12.0</td>
      <td>8.00</td>
      <td>12.0</td>
      <td>8.0</td>
      <td>12.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7070 Saint Barbara Boulevard</td>
      <td>qnpvw-uQyRn9nlClWFK9aA</td>
      <td>Mississauga</td>
      <td>1</td>
      <td>43.639236</td>
      <td>-79.716199</td>
      <td>Wild Wing</td>
      <td>Meadowvale Village</td>
      <td>L5W 0E6</td>
      <td>6</td>
      <td>2.5</td>
      <td>ON</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4502 East Towne Blvd</td>
      <td>TXiEgINSZ75d3EtvLvkc4Q</td>
      <td>Madison</td>
      <td>1</td>
      <td>43.128034</td>
      <td>-89.307157</td>
      <td>Red Lobster</td>
      <td></td>
      <td>53704</td>
      <td>45</td>
      <td>3.0</td>
      <td>WI</td>
      <td>23.0</td>
      <td>11.0</td>
      <td>22.0</td>
      <td>11.0</td>
      <td>23.0</td>
      <td>11.00</td>
      <td>22.0</td>
      <td>11.00</td>
      <td>22.0</td>
      <td>11.0</td>
      <td>22.0</td>
      <td>11.0</td>
      <td>22.0</td>
      <td>11.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1794 Liverpool Road</td>
      <td>KW4y7uDGjVfU3ClkEjIGhg</td>
      <td>Pickering</td>
      <td>1</td>
      <td>43.834351</td>
      <td>-79.090135</td>
      <td>The Works</td>
      <td></td>
      <td>L1V 1V9</td>
      <td>41</td>
      <td>3.0</td>
      <td>ON</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>30 High Tech Rd</td>
      <td>reWc1g65PNZnKz_Ub9QKOQ</td>
      <td>Richmond Hill</td>
      <td>1</td>
      <td>43.841993</td>
      <td>-79.429343</td>
      <td>Milestones Restaurants</td>
      <td></td>
      <td>L4B 4L9</td>
      <td>51</td>
      <td>2.5</td>
      <td>ON</td>
      <td>25.0</td>
      <td>11.0</td>
      <td>23.0</td>
      <td>11.0</td>
      <td>25.0</td>
      <td>10.00</td>
      <td>23.0</td>
      <td>10.00</td>
      <td>24.0</td>
      <td>11.0</td>
      <td>23.0</td>
      <td>11.0</td>
      <td>24.0</td>
      <td>11.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>280 W Beaver Creek Road, Unit 30</td>
      <td>L1XHTn7S-6har9UGAPjcWQ</td>
      <td>Richmond Hill</td>
      <td>1</td>
      <td>43.843475</td>
      <td>-79.387686</td>
      <td>Papa Chang's Tea Bistro</td>
      <td></td>
      <td>L4B 3Z1</td>
      <td>4</td>
      <td>4.0</td>
      <td>ON</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2259 Deming Way</td>
      <td>PV9CdNFDOX4_zWm3Sy3W8g</td>
      <td>Middleton</td>
      <td>1</td>
      <td>43.097806</td>
      <td>-89.519217</td>
      <td>Quaker Steak &amp; Lube</td>
      <td></td>
      <td>53562</td>
      <td>117</td>
      <td>3.0</td>
      <td>WI</td>
      <td>24.0</td>
      <td>11.0</td>
      <td>23.0</td>
      <td>11.0</td>
      <td>24.0</td>
      <td>10.00</td>
      <td>22.0</td>
      <td>10.00</td>
      <td>23.0</td>
      <td>11.0</td>
      <td>23.0</td>
      <td>11.0</td>
      <td>23.0</td>
      <td>11.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>25 W Warner Rd</td>
      <td>HRFJlSAP_EBU_MpPPmpUDQ</td>
      <td>Chandler</td>
      <td>1</td>
      <td>33.335058</td>
      <td>-111.843076</td>
      <td>Domino's Pizza</td>
      <td></td>
      <td>85225</td>
      <td>20</td>
      <td>2.5</td>
      <td>AZ</td>
      <td>26.0</td>
      <td>10.0</td>
      <td>25.0</td>
      <td>10.0</td>
      <td>26.0</td>
      <td>10.00</td>
      <td>25.0</td>
      <td>10.00</td>
      <td>25.0</td>
      <td>10.0</td>
      <td>25.0</td>
      <td>10.0</td>
      <td>25.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>106 Rue McGill</td>
      <td>58APdML-PG_OD4El2ePTvw</td>
      <td>Montréal</td>
      <td>1</td>
      <td>45.499409</td>
      <td>-73.555200</td>
      <td>Le Cartet</td>
      <td>Ville-Marie</td>
      <td>H2Y 2E5</td>
      <td>344</td>
      <td>4.0</td>
      <td>QC</td>
      <td>19.5</td>
      <td>7.0</td>
      <td>19.5</td>
      <td>7.0</td>
      <td>16.0</td>
      <td>9.00</td>
      <td>16.0</td>
      <td>9.00</td>
      <td>19.5</td>
      <td>7.0</td>
      <td>19.5</td>
      <td>7.0</td>
      <td>19.5</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1675 Lake Shore Boulevard E</td>
      <td>Z1r6b30Tg0n0ME4-Zj2wQQ</td>
      <td>Toronto</td>
      <td>1</td>
      <td>43.663010</td>
      <td>-79.310898</td>
      <td>Boardwalk Place</td>
      <td></td>
      <td>M4W 3L6</td>
      <td>13</td>
      <td>3.0</td>
      <td>ON</td>
      <td>16.0</td>
      <td>8.0</td>
      <td>16.0</td>
      <td>8.0</td>
      <td>16.0</td>
      <td>8.00</td>
      <td>16.0</td>
      <td>8.00</td>
      <td>16.0</td>
      <td>8.0</td>
      <td>16.0</td>
      <td>8.0</td>
      <td>16.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2400 E Lake Mead Blvd</td>
      <td>LDMCrFlGIFUN6L-FEFgzWg</td>
      <td>Las Vegas</td>
      <td>1</td>
      <td>36.196203</td>
      <td>-115.116799</td>
      <td>El Pollo Loco</td>
      <td></td>
      <td>89030</td>
      <td>12</td>
      <td>3.0</td>
      <td>NV</td>
      <td>23.0</td>
      <td>9.0</td>
      <td>23.0</td>
      <td>9.0</td>
      <td>23.0</td>
      <td>9.00</td>
      <td>23.0</td>
      <td>9.00</td>
      <td>23.0</td>
      <td>9.0</td>
      <td>23.0</td>
      <td>9.0</td>
      <td>23.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>19 Elm Row</td>
      <td>cBHMUESPj4SNs65Xv6xWRA</td>
      <td>Edinburgh</td>
      <td>1</td>
      <td>55.950268</td>
      <td>-3.207114</td>
      <td>Valvona &amp; Crolla</td>
      <td>New Town</td>
      <td>EH2 2YZ</td>
      <td>60</td>
      <td>4.0</td>
      <td>EDH</td>
      <td>18.0</td>
      <td>9.5</td>
      <td>18.0</td>
      <td>9.5</td>
      <td>18.0</td>
      <td>9.00</td>
      <td>17.5</td>
      <td>10.00</td>
      <td>19.0</td>
      <td>9.5</td>
      <td>18.0</td>
      <td>9.5</td>
      <td>18.0</td>
      <td>9.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2641 N 44th St, Ste 100</td>
      <td>01xXe2m_z048W5gcBFpoJA</td>
      <td>Phoenix</td>
      <td>1</td>
      <td>33.478043</td>
      <td>-111.986370</td>
      <td>Five Guys</td>
      <td></td>
      <td>85008</td>
      <td>63</td>
      <td>3.5</td>
      <td>AZ</td>
      <td>22.0</td>
      <td>10.0</td>
      <td>22.0</td>
      <td>10.0</td>
      <td>22.0</td>
      <td>10.00</td>
      <td>22.0</td>
      <td>10.00</td>
      <td>22.0</td>
      <td>10.0</td>
      <td>22.0</td>
      <td>10.0</td>
      <td>22.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1504 Rue Sherbrooke Ouest</td>
      <td>dEmNOTm8Rmm9JYZdGX_Lhw</td>
      <td>Montreal</td>
      <td>1</td>
      <td>45.497017</td>
      <td>-73.581008</td>
      <td>Ristorante Beatrice</td>
      <td>Ville-Marie</td>
      <td>H3G 1L3</td>
      <td>35</td>
      <td>4.0</td>
      <td>QC</td>
      <td>15.0</td>
      <td>12.0</td>
      <td>23.0</td>
      <td>18.0</td>
      <td>23.0</td>
      <td>18.00</td>
      <td>23.0</td>
      <td>18.00</td>
      <td>15.0</td>
      <td>12.0</td>
      <td>15.0</td>
      <td>12.0</td>
      <td>15.0</td>
      <td>12.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1 S State St</td>
      <td>Bl7Y-ATTzXytQnCceg5k6w</td>
      <td>Painesville</td>
      <td>1</td>
      <td>41.726029</td>
      <td>-81.240943</td>
      <td>Sidewalk Cafe Painesville</td>
      <td></td>
      <td>44077</td>
      <td>26</td>
      <td>3.0</td>
      <td>OH</td>
      <td>16.5</td>
      <td>6.0</td>
      <td>16.5</td>
      <td>6.0</td>
      <td>16.5</td>
      <td>6.00</td>
      <td>16.5</td>
      <td>6.00</td>
      <td>16.5</td>
      <td>6.0</td>
      <td>16.5</td>
      <td>6.0</td>
      <td>16.5</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>278 Lakeshore Road E</td>
      <td>BvCHyg0GtxA6XKmRC0cQsg</td>
      <td>Mississauga</td>
      <td>1</td>
      <td>43.560591</td>
      <td>-79.576305</td>
      <td>The Great Canadian Pizza Company</td>
      <td>Port Credit</td>
      <td>L5G 1H1</td>
      <td>18</td>
      <td>4.0</td>
      <td>ON</td>
      <td>22.0</td>
      <td>11.0</td>
      <td>22.0</td>
      <td>11.0</td>
      <td>22.0</td>
      <td>11.00</td>
      <td>22.0</td>
      <td>11.00</td>
      <td>22.0</td>
      <td>11.0</td>
      <td>22.0</td>
      <td>11.0</td>
      <td>22.0</td>
      <td>11.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2009 Kenyon Rd</td>
      <td>fl2TPNWrchkCbNEg0utjvw</td>
      <td>Urbana</td>
      <td>1</td>
      <td>40.133197</td>
      <td>-88.198577</td>
      <td>Steak 'n Shake</td>
      <td></td>
      <td>61802</td>
      <td>14</td>
      <td>2.0</td>
      <td>IL</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Gartloch Road</td>
      <td>b5VIe-VnkOOwiwkjGw071A</td>
      <td>Glasgow</td>
      <td>1</td>
      <td>55.934442</td>
      <td>-3.105057</td>
      <td>McChans Oriental Express</td>
      <td></td>
      <td>G33 5AL</td>
      <td>3</td>
      <td>4.0</td>
      <td>EDH</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>6325 E Russell Rd</td>
      <td>DiA78qPtp6rfRNdomzjBbw</td>
      <td>Las Vegas</td>
      <td>1</td>
      <td>36.083384</td>
      <td>-115.033995</td>
      <td>Joshan Filipino Oriental Market</td>
      <td>Southeast</td>
      <td>89122</td>
      <td>17</td>
      <td>3.5</td>
      <td>NV</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1949 W Ray Rd, Ste 235</td>
      <td>d2fkRF67jiASrXxHfbmJuA</td>
      <td>Chandler</td>
      <td>1</td>
      <td>33.319230</td>
      <td>-111.874147</td>
      <td>Jade Palace</td>
      <td></td>
      <td>85224</td>
      <td>3</td>
      <td>2.5</td>
      <td>AZ</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>4323 W Cactus Rd</td>
      <td>wsyNO9Ac4gqGYTBfNeM1VA</td>
      <td>Glendale</td>
      <td>1</td>
      <td>33.595849</td>
      <td>-112.152488</td>
      <td>Don Ruben's Mexican Restaurant</td>
      <td></td>
      <td>85304</td>
      <td>186</td>
      <td>4.5</td>
      <td>AZ</td>
      <td>21.0</td>
      <td>11.0</td>
      <td>21.0</td>
      <td>11.0</td>
      <td>21.0</td>
      <td>11.00</td>
      <td>20.0</td>
      <td>11.00</td>
      <td>21.0</td>
      <td>11.0</td>
      <td>21.0</td>
      <td>11.0</td>
      <td>21.0</td>
      <td>11.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>67 Mayfield Road</td>
      <td>lhryYODlAmzQLZGkwmZ8wA</td>
      <td>Edinburgh</td>
      <td>1</td>
      <td>55.930289</td>
      <td>-3.175946</td>
      <td>Mayfield Village</td>
      <td></td>
      <td>EH9 3AA</td>
      <td>3</td>
      <td>3.5</td>
      <td>EDH</td>
      <td>22.5</td>
      <td>16.5</td>
      <td>22.0</td>
      <td>16.5</td>
      <td>22.5</td>
      <td>16.50</td>
      <td>22.0</td>
      <td>16.50</td>
      <td>22.0</td>
      <td>16.5</td>
      <td>22.0</td>
      <td>16.5</td>
      <td>22.0</td>
      <td>16.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1203 E Charleston Blvd, Ste 140</td>
      <td>YTqtM2WFhcMZGeAGA08Cfg</td>
      <td>Las Vegas</td>
      <td>1</td>
      <td>36.159363</td>
      <td>-115.135949</td>
      <td>Mariscos Playa Escondida</td>
      <td>Downtown</td>
      <td>89104</td>
      <td>330</td>
      <td>4.5</td>
      <td>NV</td>
      <td>23.0</td>
      <td>10.5</td>
      <td>21.0</td>
      <td>10.5</td>
      <td>23.0</td>
      <td>10.25</td>
      <td>21.0</td>
      <td>10.25</td>
      <td>21.0</td>
      <td>10.5</td>
      <td>21.0</td>
      <td>10.5</td>
      <td>21.0</td>
      <td>10.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>South Point Hotel &amp; Casino, 9777 S Las Vegas Blvd</td>
      <td>Oto60yDwk1z72WmfWEYrjg</td>
      <td>Las Vegas</td>
      <td>1</td>
      <td>36.012191</td>
      <td>-115.173993</td>
      <td>Baja Miguel's</td>
      <td>Southeast</td>
      <td>89183</td>
      <td>175</td>
      <td>3.0</td>
      <td>NV</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>23.0</td>
      <td>11.00</td>
      <td>23.0</td>
      <td>11.00</td>
      <td>23.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>23.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>3306A N Davidson St</td>
      <td>5GJ60TyviQnyg2257CAGuA</td>
      <td>Charlotte</td>
      <td>1</td>
      <td>35.247799</td>
      <td>-80.804048</td>
      <td>FūD at Salud</td>
      <td>NoDa</td>
      <td>28205</td>
      <td>71</td>
      <td>4.5</td>
      <td>NC</td>
      <td>24.0</td>
      <td>12.0</td>
      <td>22.0</td>
      <td>12.0</td>
      <td>24.0</td>
      <td>11.00</td>
      <td>19.0</td>
      <td>12.00</td>
      <td>22.0</td>
      <td>12.0</td>
      <td>22.0</td>
      <td>12.0</td>
      <td>22.0</td>
      <td>12.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>7117 E 6th Ave</td>
      <td>wJY74R0zAgjxvBf-d4gm9g</td>
      <td>Scottsdale</td>
      <td>1</td>
      <td>33.498722</td>
      <td>-111.927451</td>
      <td>Kelly's At Southbridge</td>
      <td></td>
      <td>85251</td>
      <td>224</td>
      <td>3.5</td>
      <td>AZ</td>
      <td>26.0</td>
      <td>16.0</td>
      <td>22.0</td>
      <td>11.0</td>
      <td>26.0</td>
      <td>11.00</td>
      <td>22.0</td>
      <td>11.00</td>
      <td>23.0</td>
      <td>16.0</td>
      <td>22.0</td>
      <td>16.0</td>
      <td>22.0</td>
      <td>16.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>5445 Yonge Street</td>
      <td>iMoFE2g4kDG4FfKLJvk3Jw</td>
      <td>North York</td>
      <td>1</td>
      <td>43.777044</td>
      <td>-79.414544</td>
      <td>Buk Chang Dong Soon Tofu</td>
      <td>Willowdale</td>
      <td>M2N 5S1</td>
      <td>267</td>
      <td>4.0</td>
      <td>ON</td>
      <td>22.5</td>
      <td>11.0</td>
      <td>22.5</td>
      <td>11.0</td>
      <td>22.5</td>
      <td>11.00</td>
      <td>22.0</td>
      <td>11.00</td>
      <td>22.5</td>
      <td>11.0</td>
      <td>22.5</td>
      <td>11.0</td>
      <td>22.5</td>
      <td>11.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>10430 Northfield Rd</td>
      <td>7HFRdxVttyY9GiMpywhhYw</td>
      <td>Northfield</td>
      <td>1</td>
      <td>41.342763</td>
      <td>-81.529281</td>
      <td>Zeppe's Pizzeria</td>
      <td></td>
      <td>44067</td>
      <td>7</td>
      <td>3.0</td>
      <td>OH</td>
      <td>24.0</td>
      <td>11.0</td>
      <td>22.0</td>
      <td>11.0</td>
      <td>24.0</td>
      <td>11.00</td>
      <td>22.0</td>
      <td>12.00</td>
      <td>23.0</td>
      <td>11.0</td>
      <td>22.0</td>
      <td>11.0</td>
      <td>22.0</td>
      <td>11.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>38627</th>
      <td>50 E Valhalla Dr</td>
      <td>JHTzeOsZse-g3xkUM20nxQ</td>
      <td>Markham</td>
      <td>1</td>
      <td>43.848246</td>
      <td>-79.364146</td>
      <td>Tivoli Garden</td>
      <td>Brown's Corners</td>
      <td>L3R 0A3</td>
      <td>4</td>
      <td>3.0</td>
      <td>ON</td>
      <td>23.0</td>
      <td>6.5</td>
      <td>23.0</td>
      <td>6.5</td>
      <td>23.0</td>
      <td>6.50</td>
      <td>23.0</td>
      <td>6.50</td>
      <td>23.5</td>
      <td>6.5</td>
      <td>23.0</td>
      <td>18.5</td>
      <td>23.0</td>
      <td>6.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>38628</th>
      <td>14815 Ballantyne Village Way, Fl 2nd ,Ste 250</td>
      <td>o2O4qWlo4O0aI5oEZsUHBQ</td>
      <td>Charlotte</td>
      <td>1</td>
      <td>35.053472</td>
      <td>-80.851912</td>
      <td>Jade Asian Fusion</td>
      <td>Ballantyne</td>
      <td>28277</td>
      <td>118</td>
      <td>3.0</td>
      <td>NC</td>
      <td>23.0</td>
      <td>17.0</td>
      <td>22.0</td>
      <td>17.0</td>
      <td>14.5</td>
      <td>11.50</td>
      <td>22.0</td>
      <td>17.00</td>
      <td>22.0</td>
      <td>17.0</td>
      <td>22.0</td>
      <td>17.0</td>
      <td>22.0</td>
      <td>17.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>38629</th>
      <td>1601 E Bell Rd, Ste A-8</td>
      <td>rLTa-PztQLafDf70aKrNOA</td>
      <td>Phoenix</td>
      <td>1</td>
      <td>33.640007</td>
      <td>-112.047222</td>
      <td>Gil's Taste of Taos</td>
      <td></td>
      <td>85022</td>
      <td>68</td>
      <td>3.5</td>
      <td>AZ</td>
      <td>20.0</td>
      <td>17.0</td>
      <td>17.0</td>
      <td>9.0</td>
      <td>20.0</td>
      <td>17.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>20.0</td>
      <td>17.0</td>
      <td>15.0</td>
      <td>11.0</td>
      <td>15.0</td>
      <td>11.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>38630</th>
      <td>1813 E Baseline Rd</td>
      <td>UBj-25LU5dxJ9meb-TPc9w</td>
      <td>Tempe</td>
      <td>1</td>
      <td>33.377432</td>
      <td>-111.906919</td>
      <td>Little India</td>
      <td></td>
      <td>85281</td>
      <td>97</td>
      <td>4.0</td>
      <td>AZ</td>
      <td>21.0</td>
      <td>11.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>21.0</td>
      <td>11.00</td>
      <td>20.0</td>
      <td>11.00</td>
      <td>20.0</td>
      <td>11.0</td>
      <td>20.0</td>
      <td>11.0</td>
      <td>20.0</td>
      <td>11.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>38631</th>
      <td>2549 Yonge Street</td>
      <td>hUGsHeHHPfJ4gpj4LwHyXw</td>
      <td>Toronto</td>
      <td>1</td>
      <td>43.713297</td>
      <td>-79.399421</td>
      <td>Classico Louie's Pizzeria</td>
      <td>Yonge and Eglinton</td>
      <td>M4P 2H9</td>
      <td>33</td>
      <td>3.5</td>
      <td>ON</td>
      <td>24.0</td>
      <td>11.0</td>
      <td>24.0</td>
      <td>11.0</td>
      <td>24.0</td>
      <td>11.00</td>
      <td>23.0</td>
      <td>11.00</td>
      <td>24.0</td>
      <td>11.0</td>
      <td>24.0</td>
      <td>11.0</td>
      <td>24.0</td>
      <td>11.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>38632</th>
      <td>The Palazzo, 3255 S Las Vegas Blvd</td>
      <td>VsewHMsfj1Mgsl2i_hio7w</td>
      <td>Las Vegas</td>
      <td>1</td>
      <td>36.124434</td>
      <td>-115.169069</td>
      <td>LAVO Italian Restaurant &amp; Lounge</td>
      <td>The Strip</td>
      <td>89109</td>
      <td>1421</td>
      <td>3.5</td>
      <td>NV</td>
      <td>23.0</td>
      <td>17.0</td>
      <td>23.0</td>
      <td>17.0</td>
      <td>23.0</td>
      <td>19.00</td>
      <td>23.0</td>
      <td>10.00</td>
      <td>23.0</td>
      <td>17.0</td>
      <td>23.0</td>
      <td>17.0</td>
      <td>23.0</td>
      <td>17.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>38633</th>
      <td>7 Hunter Square, Old Town</td>
      <td>4GIqWxEvczRvgDuZ404dWw</td>
      <td>Edinburgh</td>
      <td>1</td>
      <td>55.949496</td>
      <td>-3.187993</td>
      <td>The Advocate</td>
      <td>Old Town</td>
      <td>EH1 1QW</td>
      <td>28</td>
      <td>3.5</td>
      <td>EDH</td>
      <td>23.0</td>
      <td>11.5</td>
      <td>23.0</td>
      <td>11.5</td>
      <td>23.0</td>
      <td>11.50</td>
      <td>23.0</td>
      <td>11.50</td>
      <td>23.0</td>
      <td>11.5</td>
      <td>23.0</td>
      <td>11.5</td>
      <td>23.0</td>
      <td>11.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>38634</th>
      <td>2025 Avenue Union</td>
      <td>ODZLMTbjCnpDNkW1JbMjlQ</td>
      <td>Montréal</td>
      <td>1</td>
      <td>45.504755</td>
      <td>-73.571224</td>
      <td>Thaiphon</td>
      <td>Ville-Marie</td>
      <td>H3A 0A3</td>
      <td>6</td>
      <td>2.5</td>
      <td>QC</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>38635</th>
      <td>6060 Rue Sherbrooke E</td>
      <td>kWDAdT4m3vbnmE0CgLs4gA</td>
      <td>Montréal</td>
      <td>1</td>
      <td>45.577508</td>
      <td>-73.545944</td>
      <td>Thaïzone</td>
      <td>Mercier-Hochelaga-Maisonneuve</td>
      <td>H1N 1C1</td>
      <td>5</td>
      <td>1.5</td>
      <td>QC</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>38636</th>
      <td>2026 Rue Wellington</td>
      <td>rofWaZTIuaedAxT_UKleSw</td>
      <td>Montréal</td>
      <td>1</td>
      <td>45.478815</td>
      <td>-73.558896</td>
      <td>Boom J's Cuisine</td>
      <td>Sud-Ouest</td>
      <td>H3K 1W7</td>
      <td>21</td>
      <td>4.0</td>
      <td>QC</td>
      <td>21.0</td>
      <td>11.5</td>
      <td>21.0</td>
      <td>11.5</td>
      <td>21.0</td>
      <td>11.50</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>21.0</td>
      <td>11.5</td>
      <td>21.0</td>
      <td>11.5</td>
      <td>21.0</td>
      <td>11.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>38637</th>
      <td>12001 Ontario 400</td>
      <td>FUQLZ8nAnlXJKVpAA4wvzA</td>
      <td>Vaughan</td>
      <td>1</td>
      <td>43.896377</td>
      <td>-79.558142</td>
      <td>OnRoute - King City</td>
      <td></td>
      <td></td>
      <td>9</td>
      <td>3.5</td>
      <td>ON</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>38638</th>
      <td>1947 W Broadway Rd</td>
      <td>A7gVn077Eb2B-yIKunLVpw</td>
      <td>Mesa</td>
      <td>1</td>
      <td>33.407126</td>
      <td>-111.873495</td>
      <td>Whataburger</td>
      <td></td>
      <td>85202</td>
      <td>35</td>
      <td>3.5</td>
      <td>AZ</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>38639</th>
      <td>2790 E Flamingo Rd, Ste A</td>
      <td>jLTJw1Gm9Q9KG-iS2PWkpA</td>
      <td>Las Vegas</td>
      <td>1</td>
      <td>36.115273</td>
      <td>-115.112276</td>
      <td>American Gypsy Cafe</td>
      <td>Eastside</td>
      <td>89121</td>
      <td>34</td>
      <td>4.5</td>
      <td>NV</td>
      <td>23.0</td>
      <td>11.0</td>
      <td>22.0</td>
      <td>11.0</td>
      <td>23.0</td>
      <td>11.00</td>
      <td>22.0</td>
      <td>11.00</td>
      <td>22.0</td>
      <td>11.0</td>
      <td>22.0</td>
      <td>11.0</td>
      <td>22.0</td>
      <td>11.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>38640</th>
      <td>3049 S Las Vegas Blvd</td>
      <td>RwMLuOkImBIqqYj4SSKSPg</td>
      <td>Las Vegas</td>
      <td>1</td>
      <td>36.131594</td>
      <td>-115.164767</td>
      <td>Tacos El Gordo</td>
      <td>The Strip</td>
      <td>89109</td>
      <td>2185</td>
      <td>4.0</td>
      <td>NV</td>
      <td>28.0</td>
      <td>10.0</td>
      <td>26.0</td>
      <td>10.0</td>
      <td>28.0</td>
      <td>10.00</td>
      <td>26.0</td>
      <td>10.00</td>
      <td>26.0</td>
      <td>10.0</td>
      <td>26.0</td>
      <td>10.0</td>
      <td>26.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>38641</th>
      <td>6221 Saltsburg Rd</td>
      <td>beb_QLcQySKwYFPyVu6eJw</td>
      <td>Penn Hills</td>
      <td>1</td>
      <td>40.484696</td>
      <td>-79.815871</td>
      <td>Pasquale's Pizzeria</td>
      <td></td>
      <td>15235</td>
      <td>5</td>
      <td>4.0</td>
      <td>PA</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>38642</th>
      <td>5661 Steeles Avenue  E</td>
      <td>wssONJTv6MEui3ShSzGzlw</td>
      <td>Toronto</td>
      <td>1</td>
      <td>43.833185</td>
      <td>-79.266110</td>
      <td>New Korea Restaurant</td>
      <td>Scarborough</td>
      <td>M1V 5P6</td>
      <td>7</td>
      <td>3.5</td>
      <td>ON</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>38643</th>
      <td>24961 Detroit Rd</td>
      <td>ZyGpQ9k1D0c4xd8k7I_RXw</td>
      <td>Westlake</td>
      <td>1</td>
      <td>41.470029</td>
      <td>-81.897178</td>
      <td>Lehman's Deli</td>
      <td></td>
      <td>44145</td>
      <td>42</td>
      <td>4.0</td>
      <td>OH</td>
      <td>20.0</td>
      <td>8.0</td>
      <td>20.0</td>
      <td>8.0</td>
      <td>16.0</td>
      <td>8.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>20.0</td>
      <td>8.0</td>
      <td>20.0</td>
      <td>8.0</td>
      <td>20.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>38644</th>
      <td>505 N Main St</td>
      <td>qbCPF1Z-Dher2MYUYFCX4Q</td>
      <td>Belmont</td>
      <td>1</td>
      <td>35.251733</td>
      <td>-81.043172</td>
      <td>McDonald's</td>
      <td></td>
      <td>28012</td>
      <td>16</td>
      <td>2.0</td>
      <td>NC</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>38645</th>
      <td>9970 Highway 48</td>
      <td>-0T0jfPnuBRdpNTXpOQZcA</td>
      <td>Markham</td>
      <td>1</td>
      <td>43.908612</td>
      <td>-79.268446</td>
      <td>Tim Hortons</td>
      <td></td>
      <td>L3P 3J3</td>
      <td>4</td>
      <td>2.0</td>
      <td>ON</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>38646</th>
      <td>10620 S Eastern Ave</td>
      <td>ee8aiHC6zaY9JoLUVUSc1w</td>
      <td>Henderson</td>
      <td>1</td>
      <td>35.998540</td>
      <td>-115.103376</td>
      <td>Winchell's Pub &amp; Grill</td>
      <td>Anthem</td>
      <td>89052</td>
      <td>48</td>
      <td>4.0</td>
      <td>NV</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>38647</th>
      <td>10050 W Bell Rd, Ste 8</td>
      <td>uytrX0s6etYPCcMVC07KJw</td>
      <td>Sun City</td>
      <td>1</td>
      <td>33.639496</td>
      <td>-112.276363</td>
      <td>Jade Star</td>
      <td></td>
      <td>85351</td>
      <td>34</td>
      <td>3.5</td>
      <td>AZ</td>
      <td>21.0</td>
      <td>11.0</td>
      <td>21.0</td>
      <td>11.0</td>
      <td>21.0</td>
      <td>12.00</td>
      <td>21.0</td>
      <td>12.00</td>
      <td>21.0</td>
      <td>11.0</td>
      <td>21.0</td>
      <td>11.0</td>
      <td>21.0</td>
      <td>11.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>38648</th>
      <td>820 St-Clair Avenue W</td>
      <td>NPQE0JvwjmFca83ZuAXutQ</td>
      <td>Toronto</td>
      <td>1</td>
      <td>43.680709</td>
      <td>-79.430740</td>
      <td>Stazione</td>
      <td>Wychwood</td>
      <td>M6C 1B6</td>
      <td>16</td>
      <td>3.5</td>
      <td>ON</td>
      <td>23.0</td>
      <td>17.0</td>
      <td>23.0</td>
      <td>17.0</td>
      <td>23.0</td>
      <td>17.00</td>
      <td>23.0</td>
      <td>17.00</td>
      <td>23.0</td>
      <td>17.0</td>
      <td>23.0</td>
      <td>17.0</td>
      <td>23.0</td>
      <td>17.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>38649</th>
      <td>11100 Monroe Rd</td>
      <td>Mkn4eaV7pVcat5PQ0U-4Qw</td>
      <td>Matthews</td>
      <td>1</td>
      <td>35.123349</td>
      <td>-80.729626</td>
      <td>Stacks Kitchen</td>
      <td>Arboretum</td>
      <td>28105</td>
      <td>176</td>
      <td>4.0</td>
      <td>NC</td>
      <td>15.0</td>
      <td>6.0</td>
      <td>15.0</td>
      <td>6.0</td>
      <td>15.0</td>
      <td>6.00</td>
      <td>15.0</td>
      <td>6.00</td>
      <td>15.0</td>
      <td>6.0</td>
      <td>15.0</td>
      <td>6.0</td>
      <td>15.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>38650</th>
      <td>1800 Las Vegas Blvd S</td>
      <td>HUKbH7r9TBJPri6LJbBKtw</td>
      <td>Las Vegas</td>
      <td>1</td>
      <td>36.150021</td>
      <td>-115.153717</td>
      <td>Tacos Mexico</td>
      <td></td>
      <td>89104</td>
      <td>108</td>
      <td>3.0</td>
      <td>NV</td>
      <td>9.0</td>
      <td>9.0</td>
      <td>9.0</td>
      <td>9.0</td>
      <td>9.0</td>
      <td>9.00</td>
      <td>9.0</td>
      <td>9.00</td>
      <td>9.0</td>
      <td>9.0</td>
      <td>9.0</td>
      <td>9.0</td>
      <td>9.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>38651</th>
      <td>18 Tank House Lane</td>
      <td>a8pmtlVKf7NiSLI-4KejIw</td>
      <td>Toronto</td>
      <td>1</td>
      <td>43.650741</td>
      <td>-79.358017</td>
      <td>El Catrin Destileria</td>
      <td>Distillery District</td>
      <td>M5A 3C4</td>
      <td>570</td>
      <td>3.5</td>
      <td>ON</td>
      <td>24.0</td>
      <td>11.5</td>
      <td>22.0</td>
      <td>12.0</td>
      <td>24.0</td>
      <td>10.50</td>
      <td>22.0</td>
      <td>10.50</td>
      <td>22.0</td>
      <td>12.0</td>
      <td>22.0</td>
      <td>12.0</td>
      <td>22.0</td>
      <td>12.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>38652</th>
      <td>5680 S Pecos Rd</td>
      <td>nKph91qATrPCbJ-QwZjDZw</td>
      <td>Las Vegas</td>
      <td>1</td>
      <td>36.086867</td>
      <td>-115.100505</td>
      <td>Hand Car Wash - Sinclair</td>
      <td>Southeast</td>
      <td>89120</td>
      <td>9</td>
      <td>4.5</td>
      <td>NV</td>
      <td>18.0</td>
      <td>8.0</td>
      <td>18.0</td>
      <td>8.0</td>
      <td>18.0</td>
      <td>8.00</td>
      <td>18.0</td>
      <td>8.00</td>
      <td>18.0</td>
      <td>8.0</td>
      <td>18.0</td>
      <td>8.0</td>
      <td>18.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>38653</th>
      <td>674-676 Sheppard Avenue W</td>
      <td>Ee2d2D0pjQF4oExg9PQ5tQ</td>
      <td>Toronto</td>
      <td>1</td>
      <td>43.754927</td>
      <td>-79.442695</td>
      <td>Popeyes Louisiana Kitchen</td>
      <td></td>
      <td>M3H 2S4</td>
      <td>3</td>
      <td>2.0</td>
      <td>ON</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>38654</th>
      <td>1371 Rue Sainte-Catherine E</td>
      <td>bYfEp3NMskYfEzWL8tVb4w</td>
      <td>Montréal</td>
      <td>1</td>
      <td>45.520289</td>
      <td>-73.554658</td>
      <td>Pachamama</td>
      <td>Ville-Marie</td>
      <td>H2L 2H7</td>
      <td>10</td>
      <td>4.5</td>
      <td>QC</td>
      <td>23.0</td>
      <td>11.0</td>
      <td>21.0</td>
      <td>11.0</td>
      <td>23.0</td>
      <td>11.00</td>
      <td>21.0</td>
      <td>11.00</td>
      <td>23.0</td>
      <td>11.0</td>
      <td>22.0</td>
      <td>11.0</td>
      <td>22.0</td>
      <td>11.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>38655</th>
      <td>10659 Grand Ave</td>
      <td>Pb5AfEWAB8GWlHyy-p-l1g</td>
      <td>Sun City</td>
      <td>1</td>
      <td>33.600876</td>
      <td>-112.287774</td>
      <td>Nino's Mexican Food Restaurant</td>
      <td></td>
      <td>85351</td>
      <td>50</td>
      <td>3.5</td>
      <td>AZ</td>
      <td>20.0</td>
      <td>11.0</td>
      <td>20.0</td>
      <td>11.0</td>
      <td>20.0</td>
      <td>11.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>20.0</td>
      <td>11.0</td>
      <td>20.0</td>
      <td>11.0</td>
      <td>20.0</td>
      <td>11.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>38656</th>
      <td>9335 N Tryon St, Ste 102</td>
      <td>7IQoE-EXnRCBUzaVHYlhmw</td>
      <td>Charlotte</td>
      <td>1</td>
      <td>35.313384</td>
      <td>-80.743157</td>
      <td>Papa John's Pizza</td>
      <td>University City</td>
      <td>28262</td>
      <td>9</td>
      <td>2.0</td>
      <td>NC</td>
      <td>26.0</td>
      <td>9.0</td>
      <td>25.0</td>
      <td>10.0</td>
      <td>26.0</td>
      <td>9.00</td>
      <td>24.0</td>
      <td>11.00</td>
      <td>25.0</td>
      <td>10.0</td>
      <td>25.0</td>
      <td>10.0</td>
      <td>25.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>38657 rows × 661 columns</p>
</div>





```python

```




```python

```




```python
# msk = np.random.rand(len(yelp_reviews)) < 0.7
# train_df = yelp_reviews[msk]
# test_df = yelp_reviews[~msk]

# train_df = train_df.reset_index(drop=True)
# test_dataset = test_df.reset_index(drop=True)
```




```python
# from all test data get test and validate sets
val_msk = np.random.rand(len(test_dataset)) < 0.5
test_df = test_dataset[val_msk]
valid_df = test_dataset[~val_msk]
```




```python
# X and y Train/Test data frames.
y_train = train_df.copy()['business_stars']
X_train = train_df.drop('business_stars', axis=1)
y_test = test_df.copy()['business_stars']
X_test = test_df.drop('business_stars', axis=1)
```




```python
print(train_df.dtypes)
```


    business_address          object
    business_attributes       object
    business_id               object
    business_categories       object
    business_city             object
    business_hours            object
    business_is_open           int64
    business_latitude        float64
    business_longitude       float64
    business_name             object
    business_neighborhood     object
    business_postal_code      object
    business_review_count      int64
    business_stars           float64
    business_state            object
    review_cool                int64
    review_date               object
    review_funny               int64
    review_id                 object
    review_stars               int64
    review_text               object
    review_useful              int64
    user_id                   object
    average_stars            float64
    compliment_cool            int64
    compliment_cute            int64
    compliment_funny           int64
    compliment_hot             int64
    compliment_list            int64
    compliment_more            int64
    compliment_note            int64
    compliment_photos          int64
    compliment_plain           int64
    compliment_profile         int64
    compliment_writer          int64
    cool                       int64
    elite                     object
    fans                       int64
    friends                   object
    funny                      int64
    name                      object
    review_count               int64
    useful                     int64
    yelping_since             object
    dtype: object
    



```python
# result dictionary
r2_dict = {'alpha': [], 'ridge':[], 'lasso':[]}

#List of Lambda (lol!) values
lol = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]

# Find R^2 scores for each model (Linear, Lasso, Ridge) while
# varying alpha value for Lasso and Ridge models.
for alpha in lol:
    r2_dict['alpha'].append(alpha)
    lasso = Lasso(alpha=alpha, fit_intercept=True)
    lasso.fit(X_train, y_train)
    
    lasso_preds = lasso.predict(X_test)
    r2_dict['lasso'].append(r2_score(y_test, lasso_preds))
    
    ridge = Ridge(alpha=alpha, fit_intercept=True)
    ridge.fit(X_train, y_train)
    ridge_preds = ridge.predict(X_test)
    r2_dict['ridge'].append(r2_score(y_test, ridge_preds))

# build data frame and inspect data.
r2_df = pd.DataFrame(r2_dict)
r2_df.head()
```


### EDA and revised project statement

#### 1. Description of the data

3 of the 6 yelp datasets that we are interested are consist of records:<br>
Reviews - <b>42 632 073</b> Businesses - <b>2 349 585</b> Users - <b>2 6033 964</b><br>
After finding a way to read data into dataframe (we couldn't load it by regular way), we observed data columns and corresponding types:

![](img/datashape_and_columns.png)

#### 2. Visualizations and captions that summarize the noteworthy findings of the EDA

After reading description of every field, we embarked on EDA to explore data relationship. Some of the questions we had ansewered are including but not limited by:
<ol>
<li>How many stars a business usually get? -> Most businesses have starts of 4 and 3.5. A small number of business has start that is under 1.5</li>
<li>How many reviews a business usually get? -> We can see most of the business have reviews that is less than 10, very few business have more than 300 reviews.</li>
<li>How many reviews a user usually give? -> We can see the majority of the user give less than 10 reviews to businesses</li>
<li>How does the reviews_count and star of a business related? -> Business that has more than 1000 reviews are most likey to get 2.5 to 4 stars.</li>
<li>Is there a relationship between review_count and stars? -> It seems like businesses with higher review count most likely to have 4 star rating</li>
</ol>

![](img/plot1.png)
![](img/plot2.png)

Running EDA on entire data set seemed to be a problem due to somewhat large datasets. On our fastest laptop - we could wait over 15 mins for one plot to finish. Hence, we cleaned business dataset by filtering out all businesses that are <b>closed</b>. While they can provide some interest to us, we believe that size of the population isn't a problem given the initial data size (`42.6` m). In addition, we had filtered out businesses that don't have `Restaurant` in their <u>categories</u>, since based on our project goal we are focusing just on restaurant businesses. Hence, we ended up with `579855` records which is `1.36%` from our initial input for bussiness. 

After that we have build a data frame by joining reviews with business through `business_id` (left outter join) result was then joined with user data frame by `user_id` (also left outter join). Be we joined 3 data frames, we renamed columns to be prefixed with original dataframe name. This was done to avoid confusion betwen review rating and business rating. After merged we ended up with 44 columns. Not all of those columns will be used in training data set.

To predict a rating that a user will give to a particular business, the important predictors are most likely to be `average_start` and `review_count` from the user, `stars` and `review_count` from the business. `compliment_hot`, `compliment_more` and so on from a user means how active this user is in the community, so the bigger value of those columns, the more trustworthy the review from this user would be. However, this data are not tied to one business, so  they may not be very sigificant to our variable.  

For the next (and final) milestone we are looking for building 4 models (baseline, regularized regression (Ridge), matrix factorization and either RF or ADABoost) to predict start rating for a given restaurant. We will build a simple static web page to summarize our project findings and hopefully we will have enough time to build a little demo.

#### 3. Revised project question based on the insights from EDA

After close looking at EDAs, we have decided to stick to our original question which should be formed as:

### For a given user, can we predict star rating that user will give corresponding restaurant?

Knowing this information would be very usefull to provide all sorts of recomendations for a registered user. Our goal would be to mimic existing yelp where a registered user looks for restaurant recomendations within a certain area (zip code). Yelp, isn't just listing all places filtered by highest ratings, distance or number of reviews. Instead it applies its trained model to find places (excluding paid accounts that are boosted to the top), which most likely will receive good rating and review if a given user would visit it.
