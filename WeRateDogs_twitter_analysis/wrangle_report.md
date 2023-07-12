## Reporting: wragle_report
* Create a **300-600 word written report** called "wrangle_report.pdf" or "wrangle_report.html" that briefly describes your wrangling efforts. This is to be framed as an internal document.

### Introduction
This report highlights the wrangling efforts that went into cleaning the three data sets: `twitter-archive-enhanced.csv`,`image_predictions.tsv` and `additional_data.csv`. This archive contains basic tweet data (tweet ID, timestamp, text, etc.) for all 5000+ of their tweets as they stood on August 1, 2017. The wranlging process consists of gathering, assessing and cleaning.

### 1. Data gathering

To begin, the `twitter_archive_enhanced.csv` dataset was downloaded from the WeRateDogs twitter archive.

Secondly, the `image_predictions.tsv` dataset hosted on Udacity’s servers was downloaded using the request python library. It contains a neural network's prediction of  what breed of dog is present in each tweet.
___
```python
url = 'https://d17h27t6h515a5.cloudfront.net/topher/2017/August/599fd2ad_image-predictions/image-predictions.tsv'
image_predictions = requests.get(url)
```
___

and written to a file named `image_prediction.tsv`
___
```python
with open(url.split('/')[-1].replace('-','_'),mode='wb') as file:
    file.write(image_predictions.content)
```
___
Lastly, the `additional_data.csv` dataset was queried from twitter API using tweepy pythonlibrary. To achieve this, twiter keys and tokens were used to authorize the twitter api.
___

```python
import tweepy
consumer_key    = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
consumer_secret = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
access_token    = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
access_secret   = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
```
___

The tweets are queried and dumped into a `tweet_jsons.txt` file.
''' 
___
```python
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

with open('tweet_jsons.txt','w') as file:
    for tweet_id in df_tweet1.tweet_id:
        start=time.time()
        try:
            tweet=api.get_status(tweet_id,tweet_mode='extended')
            json.dump(tweet._json,file)
            file.write('\n')
        except Exception as e:
            print(e)
        print('tweet_id: {} \t Run time: {}'.\
              format(tweet_id,time.time() -start))
```
___
After which the tweet id, retweet count and favorite count was mined and saved into a DataFrame which is then saved into a file called `additional_data.csv`
___
```python
df_list = []
with open('tweet_jsons.txt','r') as file:
    for lines in file.readlines():
        line = json.loads(lines)
        tweet_id = line["id"]
        retweet_count = line["retweet_count"]
        favorite_count = line["favorite_count"]

        df_list.append({
            'tweet_id': tweet_id,
            'retweet_count': retweet_count,
            'favorite_count': favorite_count
        })
df = pd.DataFrame(df_list,columns['tweet_id',
                                  'retweet_count',
                                  'favorite_count'])
df.to_csv('additional_data.csv',index=False)
```
___

### 2. Asessing data
The gathered data as visually assessed using Microsoft excel and programmatically assessed for quality (dirty) and tidiness (messy) issues. Dirty data has issues with its contentin areas such as completeness, validity, accuracy, and consistency.
While untidy data has issues with its structure. For data to be tidy:
1. Each variable forms a column.
2. Each observation forms a row.
3. Each type of observational unit forms a table.

THe following quality and tidiness issues were identified for all three datasets:

#### Quality issues

> ##### Twitter archives

1. `text` column contains 3 variables text, ratings and Urls instead of only text

2. In_reply_to_status_id, in_reply_to_user_id table contain to many missing values and of little value

3. Some entires are retweets as shown in `retweeted_status_id` column 

4. `retweeted_status_id`, `retweeted_status_user_id`, `retweeted_status_timestamp` columns are not useful once retweet entries are droped

5. Incorrect data type of `timestamp` column in string instead of datetime

6. `expanded_urls` column contain duplicated values

7. `source` column contains extraneous 

8. tweet_id are in different format and thus cannot be used as merge criteria

9. `name` column contain inconsistent name and 'None' format

> ##### Image prediction

1. Some entries in `p1_dog`, `p2_dog`, `p3_dog` columns table are not dog ratings

2. `p1`, `p2`, `p3` columns name are not decriptive 

3. Dog names are not consistent with some having small and others large caps 

4. `p1`, `p2`, `p3` columns in image prediction table contian inconsistent format for dog name, some with capitals whilst some with small caps 

5. tweet_id are indifferent format and thus cannot be used as merge criteria

> ##### additional data

1. tweet_id are indifferent format and thus cannot be used as merge criteria

#### Tidiness issues
> #### twitter achive

1.  `Doggo`,`floofer`, `pupper`, `puppo` in twitter archive table are one variable yet they are in separate columns

2. Contain observations beyoung beyond August 1st, 2017

> ##### additional data

1. additional data table should be merged of twitter archive and image prediction

### 3. Cleaning data
The quality and tidiness issues identified in the assessing stage are cleaned programmatically using the define, code and test process.

#### 3.1.0. quality issues
#### 3.1.1. twitter archives
##### Define: separate and remove ratings and urls from text as they are already present in adjacent columns
```python
tweet_arch_copy.text = pd.Series(tweet_arch_copy.text.\
          apply(lambda x:x.split('/10')[0][:-3]))
```
#### Define: drop columns: in_reply_to_status_id, in_reply_to_user_id in twitter archive  table as they are of little relevenace
```python
drop_col = ['in_reply_to_status_id', 'in_reply_to_user_id']
tweet_arch_copy.drop(drop_col,axis=1,inplace=True)
```
#### Define: filter out retweet entries and drop retweeted_status_id columns
```python
tweet_arch_copy = tweet_arch_copy[tweet_arch_copy.retweeted_status_id.isnull()]
```
#### Define:  drop retweeted_status_id, retweeted_status_user_id, retweeted_status_timestamp 
```python
drop_cols = ['retweeted_status_id',
             'retweeted_status_user_id',
             'retweeted_status_timestamp']
tweet_arch_copy.drop(drop_cols,axis=1,inplace=True)
```
#### Define: remove repeated ULRs
```python
tweet_arch_copy.drop_duplicates(subset='expanded_urls',keep='first')
```
#### Define: drop extraneous details from source 
```python
tweet_arch_copy.source = tweet_arch_copy.source\
                          .apply(lambda x:soup (x,'lxml')\
                          .contents[0].body.a.contents[0])
```
#### Define: convert tweet_id to int64 
```python 
tweet_arch_copy.tweet_id = tweet_arch_copy.tweet_id.astype(np.int64)
```
#### Define: convert name format to lower caps and replace None with np.nan 
```python
tweet_arch_copy.name = tweet_arch_copy.name.str.lower()
tweet_arch_copy.name.replace('none',np.nan,inplace=True)
```
#### 3.1.2. Image prediction
#### Define: filter out non-dog entries with booloean using `p1_dog`, `p2_dog`, `p3_dog` columns and drop those columns afterwards
```python
pred_drop = ['p1_dog','p2_dog','p3_dog']
img_pred_copy = img_pred_copy.query('p1_dog==True')\
                             .drop(pred_drop,axis=1)
```
#### Define: rename colums with more descriptive names
```python
replaced = {'p1':'first_class_prediction','p1_conf':'first_class_prediction_confidence',\
            'p2':'second_class_prediction','p2_conf':'second_class_prediction_confidence',\
            'p3':'third_class_prediction','p3_conf':'third_class_prediction_confidence'}
img_pred_copy.rename(columns=replaced ,inplace=True)
```
#### Define: `p1`, `p2`, `p3` columns in image prediction table contian inconsistent format for dog name, some with capitals ,others with small caps. change to lower caps
```python
img_pred_copy.first_class_prediction = img_pred_copy\
                                        .first_class_prediction\
                                        .str.replace('-','_')\
                                        .str.replace('_',' ')\
                                        .str.lower()

img_pred_copy.second_class_prediction = img_pred_copy\
                                        .second_class_prediction\
                                        .str.replace('-','_')\
                                        .str.replace('_',' ')\
                                        .str.lower()

img_pred_copy.third_class_prediction = img_pred_copy\
                                        .third_class_prediction\
                                        .str.replace('-','_')\
                                        .str.replace('_',' ')\
                                        .str.lower()
```
#### Define: convert tweet_id to int
```python
img_pred_copy.tweet_id = img_pred_copy.tweet_id.astype(np.int64)
```
#### 3.1.3. Additional data
#### Define: convert tweet_id to int
```python
add_data_copy.tweet_id = add_data_copy.tweet_id.astype(np.int64)```

### 3.2.0. tidiness issues cleanup
#### 3.2.1. twitter achives
#### Define: concatenate 'doggo','pupper', 'floofer', 'puppo' columns into one column called stage
```python

stage = ['doggo','pupper', 'floofer', 'puppo' ]
for i in stage:
    tweet_arch_copy[i] = tweet_arch_copy[i].replace('None', '')
    
tweet_arch_copy['stage'] = tweet_arch_copy.doggo.str.\
                           cat(tweet_arch_copy.floofer).str.\
                           cat(tweet_arch_copy.pupper).str.\
                           cat(tweet_arch_copy.puppo)

tweet_arch_copy.drop(columns=stage,axis=1,inplace=True)
```

#### Define: filter entries below Auhust, 1st, 2017
```python
tweet_arch_copy = tweet_arch_copy.query('timestamp < "2017-08-01"')```

#### 3.2.2. Additional data 
#### Define: merge additional data, image prediction and twitter archive 
```python
twitter_archive_master = tweet_arch_copy\
                        .merge(add_data_copy,on='tweet_id')\
                        .merge(img_pred_copy,on='tweet_id')```
                        
### 4. Storing Data
Save gathered, assessed, and cleaned master dataset to a CSV file named `"twitter_archive_master.csv"`.
```python
twitter_archive_master.to_csv('twitter_archive_masterr.csv')```


```python
from subprocess import call
call(['python', '-m', 'nbconvert', 'wrangle_report.ipynb'])
```




    0




```python

```
