#!/usr/bin/env python
# coding: utf-8

# ## Importing Dependencies

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Reading the dataset

# In[2]:


df = pd.read_csv('spam.csv', encoding='latin1')


# In[3]:


df.sample(5)


# In[4]:


df.shape


# In[5]:


## our project will composed of the following steps
# 1. Data Cleaning
# 2. EDA
# 3. Text Processing
# 4. Model Building
# 5. Evaluation
# 6. Website
# 7. Deployment


# ### 1. Data Cleaning

# In[6]:


df.info()


# In[7]:


# dropping the last three columns
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)


# In[8]:


df.head()


# In[9]:


# renaming the columns
df.rename(columns={'v1':'target','v2':'text'},inplace=True)


# In[10]:


# assigning the ham =0 and spam=1

from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()


# In[11]:


# labels (0,1) assinged 
df['target']=encoder.fit_transform(df['target'])


# In[12]:


df.head()


# In[13]:


# missing values
df.isnull().sum()


# In[14]:


# check duplicate values
df.duplicated().sum()


# In[15]:


# keeeping the first and dropping the duplicated values
df = df.drop_duplicates(keep='first')


# In[16]:


df.duplicated().sum()     # no duplicate value left


# In[17]:


df.shape


# ### 2. EDA

# In[18]:


df.head()


# In[19]:


#0=> ham   4516
#1=> spam   653

df['target'].value_counts()


# In[20]:


# for better visualization pie chart is used

# autopct=%0.2f is for percentage having 2 decimal point

plt.pie(df['target'].value_counts(), labels=['ham','spam'],autopct="%0.2f")
plt.show()


# In[21]:


# imbalnced data... 87.37% ham and 12.63% spam


# In[22]:


import nltk


# In[23]:


nltk.download('punkt')


# We will create three extra column. For counting the no. of characters, no. of words, no. of words in the message/email.

# In[24]:


df['num_characters']=df['text'].apply(len)     # gives the number of characters as length


# In[25]:


df.head()


# In[26]:


# number of words in a message
df['num_words']=df['text'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[27]:


df.head()


# In[28]:


# number of sentence in a message
df['num_sentence']=df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))


# In[29]:


df.head()


# In[30]:


df[['num_characters','num_words','num_sentence']].describe()


# In[31]:


# ham message descritption
df[df['target']==0][['num_characters','num_words','num_sentence']].describe()


# In[32]:


# spam message descritption
df[df['target']==1][['num_characters','num_words','num_sentence']].describe()


# In[33]:


import seaborn as sns


# In[34]:


#plt.figure(figsize=(12,6))
sns.histplot(df[df['target']==0]['num_characters'])
sns.histplot(df[df['target']==1]['num_characters'],color='red')


# In[35]:


#plt.figure(figsize=(12,6))
sns.histplot(df[df['target']==0]['num_words'])
sns.histplot(df[df['target']==1]['num_words'],color='red')


# In[36]:


#plt.figure(figsize=(12,6))
sns.histplot(df[df['target']==0]['num_sentence'])
sns.histplot(df[df['target']==1]['num_sentence'],color='red')


# In[37]:


sns.pairplot(df,hue='target')


# In[38]:


# Select only numeric columns
numeric_df = df.select_dtypes(include=[np.number])

# Calculate the correlation matrix
numeric_df.corr()



# In[39]:


sns.heatmap(numeric_df.corr(),annot=True)


# ### 3. Data Preprocessing
# - Lower Case
# - Tokenization
# - Removing Special  Characters
# - Removing stop words and punctuation
# - Stemming
# 

# In[40]:


import string                         # for punctuation
from nltk.corpus import stopwords    
from nltk.stem.porter import PorterStemmer


# In[41]:


ps= PorterStemmer()


# ### Text Transformation Function
# 
# The `transform_text` function processes a given text string by performing several text preprocessing steps:
# 
# 1. **Convert to Lowercase**: Transforms all characters in the text to lowercase.
# 2. **Tokenization**: Splits the text into individual words.
# 3. **Remove Non-Alphanumeric Characters**: Filters out any special characters, keeping only alphanumeric words.
# 4. **Remove Stopwords and Punctuation**: Eliminates common English stopwords and punctuation.
# 5. **Stemming**: Reduces words to their root form (e.g., "dancing" to "danc" and "loving" to "love").
# 
# The function returns the processed text as a single string, ready for further natural language processing tasks.
# 

# In[42]:


def transform_text(text):
    text = text.lower()
    # breaking into separate words
    text = nltk.word_tokenize(text)
    
    # as text is converted to list after tokenization- so useing loop
    y=[]
    for i in text:
        if i.isalnum():   # just include alphanumeric- remove special characters
            y.append(i)
            
    text=y[:]   # removing stopwords and punctuation
    y.clear()
    
    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)
            
    text=y[:]    # stemming  dancing-> danc, loving-> love
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)


# In[43]:


transform_text("Did you like my presentation in Machine lEarning")


# In[44]:


df["transformed_text"]=df["text"].apply(transform_text)


# In[45]:


df.head()


# Creating the **word clouds** for the spam and ham msgs. Using the target column and the tranformed text for this word cloud.

# In[46]:


# generating the word cloud for spam msgs
from wordcloud import WordCloud
wc=WordCloud(width=500,height=500,min_font_size=10,background_color='white')


# In[47]:


spam_wc=wc.generate(df[df['target']==1]['transformed_text'].str.cat(sep=" "))


# In[48]:


plt.figure(figsize=(15,6))
plt.imshow(spam_wc)


# In[49]:


# generating the word cloud for ham msgs
ham_wc = wc.generate(df[df['target'] == 0]['transformed_text'].str.cat(sep=" "))


# In[50]:


plt.figure(figsize=(15,6))
plt.imshow(ham_wc)


# In[51]:


spam_corpus = []
for msg in df[df['target'] == 1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)


# In[52]:


len(spam_corpus)


# In[53]:


from collections import Counter
common_words = Counter(spam_corpus).most_common(30)

# Convert to DataFrame
df_common_words = pd.DataFrame(common_words, columns=['word', 'count'])

# Create the barplot
sns.barplot(x='word', y='count', data=df_common_words)
plt.xticks(rotation='vertical')
plt.show()


# In[54]:


ham_corpus = []
for msg in df[df['target'] == 0]['transformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)


# In[55]:


len(ham_corpus)


# In[56]:


from collections import Counter
common_words = Counter(ham_corpus).most_common(30)

# Convert to DataFrame
df_common_words = pd.DataFrame(common_words, columns=['word', 'count'])

# Create the barplot
sns.barplot(x='word', y='count', data=df_common_words)
plt.xticks(rotation='vertical')
plt.show()


# ## 4. Model Building

# In[57]:


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=3000)


# In[58]:


#X = cv.fit_transform(df['transformed_text']).toarray()
X = tfidf.fit_transform(df['transformed_text']).toarray()


# In[59]:


X.shape


# In[60]:


y = df['target'].values


# In[61]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)


# We will use different navie's byes ML algorithm beacuse in order to know the data distribution. So using three of them and to observe the accuracy and precision for all. 

# In[62]:


from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score


# In[63]:


gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()


# **Guassion Naive Bayes**

# In[64]:


gnb.fit(X_train,y_train)
y_pred1 = gnb.predict(X_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))


# **Multi-nomial Naive Bayes**

# In[65]:


mnb.fit(X_train,y_train)
y_pred2 = mnb.predict(X_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))


# **Bernoulli Naive Bayes**

# In[66]:


bnb.fit(X_train,y_train)
y_pred3 = bnb.predict(X_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))


# - due to good precision of multinomial naive bayes we will use multinomial naive bayes with tfidfVectorizer 

# ## 5. Pickling the files for the model deployment

# As our multinomial naive bayes gives us good accuracy and precision so files from this model will be pickled.

# In[68]:


import pickle
pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))


# In[ ]:




