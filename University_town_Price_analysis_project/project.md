---
> # <span style="Snell Roundhan"> ``University towns or residential towns ?``  </span>
> ### <span style="Snell Roundhan"> ``Data-driven decision making for a real estate firm``  </span>
---
**Problem definition**

 <justify> Real estate companies, like all other industries, face significant decline in prices during recessions with certain asset classes being less affected than others. In this analysis, I investigate university towns and non-university towns for which is less affected by recession based on publicly available historic data. This is geared towards presenting top-level company executives with market insights for decision making </justify>

**Hypothesis**: University towns have their mean housing prices less effected by recessions. 

***Process*** : Wrangle data sources into clean DataFrame and run a t-test to compare the ratio of the mean price of houses in university towns the quarter before the recession starts compared to the recession bottom.

`note:`
* A _recession_ is is a period of two consecutive quarters of GDP decline, and ending with two consecutive quarters of GDP growth, where the _recession bottom_ is the quarter within a recession which had the lowest GDP.

* A _university town_ is a city which has a high percentage of university students compared to the total population of the city.

* A _quarter_ is a specific three month period, Q1 is January through March, Q2 is April through June, Q3 is July through September, Q4 is October through December.

`Data files are available`
* From the [Zillow research data site](http://www.zillow.com/research/data/) there is housing data for the United States. In particular the datafile for [all homes at a city level](http://files.zillowstatic.com/research/public/City/City_Zhvi_AllHomes.csv), ```City_Zhvi_AllHomes.csv```, has median home sale prices at a fine grained level.
* From the Wikipedia page on college towns is a list of [university towns in the United States](https://en.wikipedia.org/wiki/List_of_college_towns#College_towns_in_the_United_States) which has been copy and pasted into the file ```university_towns.txt```.
* From Bureau of Economic Analysis, US Department of Commerce, the [GDP over time](http://www.bea.gov/national/index.htm#gdp) of the United States in current dollars (use the chained value in 2009 dollars), in quarterly intervals, in the file ```gdplev.xls```. For this assignment, only look at GDP data from the first quarter of 2000 onward.



```python
import numpy as np,pandas as pd
from scipy.stats import ttest_ind
```


```python
# Copies of data files were made
!cp 'City_Zhvi_AllHomes.csv' 'housing_data.csv'
!cp 'university_towns.txt'   'univerity_towns_list.txt'
!cp 'gdplev.xls'             'GDP_data.xls'
```

> `puzzle 1`
We would open the `univesity_towns_list.txt` file which contains each state and their respective ragions. These would be organized into a more organised DataFrame.  The corresponding GDP information about these states and regions would then be added from `GDP_data.xls`. 


```python
def get_list_of_university_towns():
    '''Returns a DataFrame of towns and the states they are in from the 
    university_towns.txt list. The format of the DataFrame is like:
    DataFrame( [ ["Michigan", "Ann Arbor"], ["Michigan", "Yipsilanti"] ], 
    columns=["State", "RegionName"]  )
    
    The following cleaning are done:

    1. For "State", characters from "[" to the end are removed.
    2. For "RegionName", when applicable, every character from " (" to the end are removed.
    3. Newline character '\n' are removed as well. '''
    
    df = pd.DataFrame([],columns=['State','RegionName'])
    with open('univerity_towns_list.txt','r') as university_list:
        for line in university_list:
            if '[edit]' in line:
                State = line[:line.find('[')]
                continue
            if '(' in line:
                Town = line[:line.find('(')-1]
                df2  = pd.DataFrame([[State,Town]],columns=['State','RegionName'])
                df   = pd.concat([df,df2])    
    return df
get_list_of_university_towns().head()
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
      <th>State</th>
      <th>RegionName</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alabama</td>
      <td>Auburn</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Alabama</td>
      <td>Florence</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Alabama</td>
      <td>Jacksonville</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Alabama</td>
      <td>Livingston</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Alabama</td>
      <td>Montevallo</td>
    </tr>
  </tbody>
</table>
</div>



> `Puzzle 2` Next we read in the `GDP_data.xls`, which contains the quarterly GDP of United States in current dollars, along with a bunch of other data. However we would use the `chained value in 2009 dollars`, forming a DataFrame with colum: quarters and associated GDP.


```python
def get_GDP():
    ''' Returns a DataFrame of Quarter and GDP from the 
    GDP_data.xls
    
    The following cleaning are done:

    1. Headers and irrelevant rows are removed.
    2. Only columns relating to quarter and GDP are selected and renamed.
    3. Data is started from '2008q1' 
    '''    
    GDP = pd.read_excel('GDP_data.xls',header=None, skiprows=5)[[4,6]]\
                            .rename(columns={4:'Quarter',6:'GDP'})\
                            .dropna()\
                            .set_index('Quarter')
    return GDP.iloc[GDP.index.get_loc('2008q1'):]
get_GDP().head()
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
      <th>GDP</th>
    </tr>
    <tr>
      <th>Quarter</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2008q1</th>
      <td>14889.5</td>
    </tr>
    <tr>
      <th>2008q2</th>
      <td>14963.4</td>
    </tr>
    <tr>
      <th>2008q3</th>
      <td>14891.6</td>
    </tr>
    <tr>
      <th>2008q4</th>
      <td>14577</td>
    </tr>
    <tr>
      <th>2009q1</th>
      <td>14375</td>
    </tr>
  </tbody>
</table>
</div>



>`Puzzle 3` With the GDP DataFrame, we can figure out the start , bottom and end of the recession. Its worth noting that a recession is a period of two consecutive quarters of GDP decline, and ending with two consecutive quarters of GDP growth. The GDP bottom is the lowest GDP point within the recession. This information would be used for selecting declining housing prices shortly


```python
def get_recession_start():
    '''Returns the year and quarter of the recession start time as a 
    string value in a format such as 2005q3'''
    # a>b>c
    
    df=get_GDP().astype(np.float64)
    for i in range(0,len(df)):
        if df.iloc[i]['GDP'] < df.iloc[i-1]['GDP'] and df.iloc[i-1]['GDP'] < df.iloc[i-2]['GDP']:
            start = df.iloc[i].name
            return start
    return None

def get_recession_end():
    '''Returns the year and quarter of the recession end time as a 
    string value in a format such as 2005q3'''
    # a<b<c
    #recession ends only after starting, hence count point start at the start of recession start
    recession_start = get_recession_start() 
    df = get_GDP()
    recession_start_index = df.index.get_loc(recession_start)
    df = df.iloc[recession_start_index:]
    for i in range(0,len(df)):
        if df.iloc[i]['GDP'] > df.iloc[i-1]['GDP'] and df.iloc[i-1]['GDP'] > df.iloc[i-2]['GDP']:
            end = df.iloc[i].name
            return end
    return None

def get_recession_bottom():
    '''Returns the year and quarter of the recession bottom time as a 
    string value in a format such as 2005q3'''
    df = get_GDP()
    recession_star_inde = df.index.get_loc(get_recession_start())
    recession_end_index = df.index.get_loc(get_recession_end())
    df = df.iloc[recession_star_inde:recession_end_index+1].astype(np.float64)  #recession_end_index +1 cos last index is inclusive
    bottom = df.nsmallest(1, 'GDP').index[0]
    return bottom

get_recession_start(),get_recession_bottom(), get_recession_end()
```




    ('2008q4', '2009q2', '2009q4')



> <justify>`Puzzle 4` It gets interesting from here. We read in the `housing_data.csv` which contains the monthly housing prices of each state and associated regions of the united states. The monthly prices are aggregated into quarters with mean values.</justify>


```python
# Use this dictionary to map state names to two letter acronyms
states = {'OH': 'Ohio', 'KY': 'Kentucky', 'AS': 'American Samoa', 
          'NV': 'Nevada', 'WY': 'Wyoming', 'NA': 'National', 
          'AL': 'Alabama','MD': 'Maryland', 'AK': 'Alaska', 'UT': 'Utah', 
          'OR': 'Oregon','MT': 'Montana', 'IL': 'Illinois', 'TN': 'Tennessee',
          'DC': 'District of Columbia','VT': 'Vermont', 'ID': 'Idaho', 'AR': 'Arkansas',
          'ME': 'Maine', 'WA': 'Washington','HI': 'Hawaii', 'WI': 'Wisconsin', 'MI': 'Michigan',
          'IN': 'Indiana', 'NJ': 'New Jersey','AZ': 'Arizona', 'GU': 'Guam', 'MS': 'Mississippi',
          'PR': 'Puerto Rico', 'NC': 'North Carolina','TX': 'Texas', 'SD': 'South Dakota', 
          'MP': 'Northern Mariana Islands', 'IA': 'Iowa', 'MO': 'Missouri','CT': 'Connecticut', 
          'WV': 'West Virginia', 'SC': 'South Carolina', 'LA': 'Louisiana', 'KS': 'Kansas', 
          'NY': 'New York', 'NE': 'Nebraska', 'OK': 'Oklahoma', 'FL': 'Florida', 'CA': 'California', 
          'CO': 'Colorado', 'PA': 'Pennsylvania', 'DE': 'Delaware', 'NM': 'New Mexico', 'RI': 'Rhode Island',
          'MN': 'Minnesota', 'VI': 'Virgin Islands', 'NH': 'New Hampshire', 'MA': 'Massachusetts', 
          'GA': 'Georgia', 'ND': 'North Dakota', 'VA': 'Virginia'}

def convert(col):
    '''
    Converts column name to quarter. e.g. '2000-1' to '2000q1'
    '''
    year,month = col.split('-')
    month = int(month)
    if month <= 3:
        return year+'q1'
    elif month <= 6:
        return year+'q2'
    elif month <= 9:
        return year+'q3'
    else:
        return year+'q4'
    return None


def convert_housing_data_to_quarters():
    '''Converts the housing data to quarters and returns it as mean 
    values in a dataframe. This dataframe is a dataframe with
    columns for 2000q1 through 2016q3, and has have a multi-index
    in the shape of ["State","RegionName"].
    
    The resulting dataframe has 67 columns, and 10,730 rows.
    '''
    
    df = pd.read_csv('housing_data.csv')\
                                        .drop(['RegionID','Metro','CountyName','SizeRank'],axis=1)\
                                        .replace({'State':states})\
                                        .replace(to_replace='NaN',value=np.nan)\
                                        .set_index(["State","RegionName"])\
                                        .astype(np.float64)

    column_index = df.columns.tolist().index('2000-01') #get index of '2000-01'
    df = df.drop(df.columns[:column_index],axis=1)      #select columns 2000-01 and above using index

    l = len(df.columns)                                 #length of column
    i = 0                                               #iterator
    while i <= l:                   
        col_name = df.iloc[:,i].name                    #select column name[i]
        quarter  = convert(col_name)                    #convert to quarter
        if i+3 < l:                                     #if 4 chosen columns < length column 
            grp_to_qtr = df.iloc[:,i:i+3]               # select those 4 columns
        else:                                           #if 4 chosen columns are beyound the column limit
            grp_to_qtr = df.iloc[:,i:l]                 # select columns to the end of column limit
        df[quarter] = grp_to_qtr.mean(axis=1)           #compute mean of constituent months of quarter into new column with corresponding column quarter name
        i+=3                                            #move to the next set of 4 months
    df = df.drop(df.columns[:l],axis=1)                 #drop the old columns
    return df

convert_housing_data_to_quarters().head()
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
      <th></th>
      <th>2000q1</th>
      <th>2000q2</th>
      <th>2000q3</th>
      <th>2000q4</th>
      <th>2001q1</th>
      <th>2001q2</th>
      <th>2001q3</th>
      <th>2001q4</th>
      <th>2002q1</th>
      <th>2002q2</th>
      <th>...</th>
      <th>2014q2</th>
      <th>2014q3</th>
      <th>2014q4</th>
      <th>2015q1</th>
      <th>2015q2</th>
      <th>2015q3</th>
      <th>2015q4</th>
      <th>2016q1</th>
      <th>2016q2</th>
      <th>2016q3</th>
    </tr>
    <tr>
      <th>State</th>
      <th>RegionName</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>New York</th>
      <th>New York</th>
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
      <td>...</td>
      <td>515466.666667</td>
      <td>522800.000000</td>
      <td>528066.666667</td>
      <td>532266.666667</td>
      <td>540800.000000</td>
      <td>557200.000000</td>
      <td>572833.333333</td>
      <td>582866.666667</td>
      <td>591633.333333</td>
      <td>587200.0</td>
    </tr>
    <tr>
      <th>California</th>
      <th>Los Angeles</th>
      <td>207066.666667</td>
      <td>214466.666667</td>
      <td>220966.666667</td>
      <td>226166.666667</td>
      <td>233000.000000</td>
      <td>239100.000000</td>
      <td>245066.666667</td>
      <td>253033.333333</td>
      <td>261966.666667</td>
      <td>272700.000000</td>
      <td>...</td>
      <td>498033.333333</td>
      <td>509066.666667</td>
      <td>518866.666667</td>
      <td>528800.000000</td>
      <td>538166.666667</td>
      <td>547266.666667</td>
      <td>557733.333333</td>
      <td>566033.333333</td>
      <td>577466.666667</td>
      <td>584050.0</td>
    </tr>
    <tr>
      <th>Illinois</th>
      <th>Chicago</th>
      <td>138400.000000</td>
      <td>143633.333333</td>
      <td>147866.666667</td>
      <td>152133.333333</td>
      <td>156933.333333</td>
      <td>161800.000000</td>
      <td>166400.000000</td>
      <td>170433.333333</td>
      <td>175500.000000</td>
      <td>177566.666667</td>
      <td>...</td>
      <td>192633.333333</td>
      <td>195766.666667</td>
      <td>201266.666667</td>
      <td>201066.666667</td>
      <td>206033.333333</td>
      <td>208300.000000</td>
      <td>207900.000000</td>
      <td>206066.666667</td>
      <td>208200.000000</td>
      <td>212000.0</td>
    </tr>
    <tr>
      <th>Pennsylvania</th>
      <th>Philadelphia</th>
      <td>53000.000000</td>
      <td>53633.333333</td>
      <td>54133.333333</td>
      <td>54700.000000</td>
      <td>55333.333333</td>
      <td>55533.333333</td>
      <td>56266.666667</td>
      <td>57533.333333</td>
      <td>59133.333333</td>
      <td>60733.333333</td>
      <td>...</td>
      <td>113733.333333</td>
      <td>115300.000000</td>
      <td>115666.666667</td>
      <td>116200.000000</td>
      <td>117966.666667</td>
      <td>121233.333333</td>
      <td>122200.000000</td>
      <td>123433.333333</td>
      <td>126933.333333</td>
      <td>128700.0</td>
    </tr>
    <tr>
      <th>Arizona</th>
      <th>Phoenix</th>
      <td>111833.333333</td>
      <td>114366.666667</td>
      <td>116000.000000</td>
      <td>117400.000000</td>
      <td>119600.000000</td>
      <td>121566.666667</td>
      <td>122700.000000</td>
      <td>124300.000000</td>
      <td>126533.333333</td>
      <td>128366.666667</td>
      <td>...</td>
      <td>164266.666667</td>
      <td>165366.666667</td>
      <td>168500.000000</td>
      <td>171533.333333</td>
      <td>174166.666667</td>
      <td>179066.666667</td>
      <td>183833.333333</td>
      <td>187900.000000</td>
      <td>191433.333333</td>
      <td>195200.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 67 columns</p>
</div>



><justify> `Final puzzle` All the chess pieces are in place for a grand win. 
>* We have under our belt: the `recession start, bottom and list of university towns`. The decline of house prices between Recession start and bottom is found from housing data as a ratio of (`price_ratio=quarter_before_recession/recession_bottom`). 
>*The `list of university towns` are used to filter out the two groups of university towns and non university towns from the ratio column.
The T-test is used to check the difference of the mean of this two group of data
    
>*`Alternate hypothesis:` there is a differnce btn decline in prices of the university town values to the non-university towns values
    
>*`Null hypothesis:` there is no differnce btn decline in prices of the university town values to the non-university towns values</justify>

* if p-value < 0.01 then we reject the null hypothesis
* if p-value > 0.01 then we cannot reject the null hypothesis 


```python
def ans(s,p):
        if p < 0.01:
            different =True
            better='university town'
            ans='Since p-value < {}, we can reject the null hypothesis. Thus, there is a {} difference and that {} have their mean housing prices less effected by recessions.'.format(p[0],different,better,)
        else:
            different=False
            better='non-university'
            ans='Since p-value > {} we cannot reject the null hypothesis. Thus, there is a {} difference and that {} have their mean housing prices less effected by recessions.'.format(p[0],difference, better)
        return ans
    
def run_ttest():
    
    '''First creates new data showing the decline or growth of housing prices
    between the recession start and the recession bottom. Then runs a ttest
    comparing the university town values to the non-university towns values, 
    return whether the alternative hypothesis (that the two groups are the same)
    is true or not as well as the p-value of the confidence. 
    
    '''

    house_prices = convert_housing_data_to_quarters()                       #get housing prices

    rece_start_col  = house_prices.columns.get_loc(get_recession_start())   #get columns location representing recession start from pricing data 
    rece_end_col    = house_prices.columns.get_loc(get_recession_end())     #get columns location representing recession end from pricing data
    rece_bottom_col = house_prices.columns.get_loc(get_recession_bottom())  #get columns locationrepresenting recession bottom from pricing data

    start_hou_decline  = house_prices.iloc[:,rece_start_col-2] # move 2 steps back to get the start of decline from the start point of recession
    hou_prices_bottom = house_prices.iloc[:,rece_bottom_col]

    # ratio of prices recession_start/ recession_bottom
    house_prices['price_ratio'] = start_hou_decline/hou_prices_bottom
    housing_decline = pd.DataFrame(house_prices['price_ratio']) 

    #out of housig decline we select that of university town houses
    list_ut = get_list_of_university_towns().set_index(['State','RegionName'])
    ut_prices_decline = pd.merge(housing_decline,list_ut,how='inner',left_index=True,right_index=True).dropna()

    #out of housig decline, we remove university town houses leaving non university town houses
    non_ut_prices_decline = housing_decline.drop(ut_prices_decline.index.tolist(),axis=0).dropna()

    s,p = ttest_ind(ut_prices_decline,non_ut_prices_decline)

    return ans(s,p)
run_ttest()

```




    'Since p-value < 0.002724063704761164, we can reject the null hypothesis. Thus, there is a True difference and that university town have their mean housing prices less effected by recessions.'




```python

```
