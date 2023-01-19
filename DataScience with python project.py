#!/usr/bin/env python
# coding: utf-8

# In[105]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import warnings
warnings.filterwarnings('ignore')
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats 
from scipy.stats import chi2_contingency


# In[106]:


df=pd.read_csv('311_Service_Requests_from_2010_to_Present.csv')
df.head()


# In[107]:


df.tail()


# In[108]:


df.shape


# In[109]:


df.describe()


# In[110]:


df.info()


# In[111]:


df['Created Date']=pd.to_datetime(df['Created Date'])
df['Closed Date']=pd.to_datetime(df['Closed Date'])


# In[112]:


df['Request_Closing_Time']=(df['Closed Date']-df['Created Date'])

Request_Closing_Time=[]
for x in (df['Closed Date']-df['Created Date']):
    close=x.total_seconds()/60
    Request_Closing_Time.append(close)
        
df['Request_Closing_Time']=Request_Closing_Time


# In[113]:


df['Agency'].unique()


# In[114]:


sns.distplot(df['Request_Closing_Time'])
plt.show()


# In[115]:


print('Total Number of Concerns:',len(df),'\n')
print('Percentage fo Requests took less than 100 hour to get solved:',round((len(df)-(df['Request_Closing_Time']>100).sum())/len(df)*100,2),'%')
print('Percentage of Requests took less than 1000 hour to get solved:',round((len(df)-(df['Request_Closing_Time']>1000).sum())/len(df)*100,2),'%')


# In[116]:


sns.distplot(df['Request_Closing_Time'])
plt.xlim((0,7000))
plt.ylim((0,0.0005))
plt.show()


# In[117]:


df['Complaint Type'].value_counts()[:10].plot(kind='bar',color='red',alpha=0.7,figsize=(15,10))
plt.show()


# In[118]:


g=sns.catplot(x='Complaint Type',y='Request_Closing_Time',data=df)
g.fig.set_figwidth(15)
g.fig.set_figheight(7)
plt.xticks(rotation=98)
plt.ylim(0,7000)
plt.show()


# In[119]:


df['Status'].value_counts().plot(kind='bar',alpha=0.6,figsize=(15,7))
plt.show()


# In[120]:


plt.figure(figsize=(12,7))
df['Borough'].value_counts().plot(kind='bar',alpha=0.7)
plt.show()


# In[121]:


for x in df['Borough'].unique():
    print('Percentage of Request from',x,'Division:',round(df['Borough']==x).sum()/len(df)*100,2)


# In[122]:


df['Location Type'].unique()


# In[123]:


pd.DataFrame(df.groupby('Location Type')['Request_Closing_Time'].mean()).sort_values('Request_Closing_Time')


# In[124]:


pd.DataFrame(df.groupby('City')['Request_Closing_Time'].mean()).sort_values('Request_Closing_Time')


# In[125]:


pd.DataFrame((df.isnull().sum()/df.shape[0]*100)).sort_values(0,ascending=False)[:20]


# In[126]:


new_df=df.loc[:,(df.isnull().sum()/df.shape[0]*100)<=50]


# In[127]:


print('Old DataFrame Shape:',df.shape)
print('New DataFrame Shape:',new_df.shape)


# In[128]:


rem=[]
for x in new_df.columns.tolist():
    if new_df[x].nunique()<=3:
        print(x+''*10+':',new_df[x].unique())
        rem.append(x)


# In[129]:


new_df.drop(rem,axis=1,inplace=True)


# In[130]:


new_df.shape


# In[131]:


rem1=["Unique Key","Incident Address","Descriptor","Street Name","Cross Street 1","Cross Street 2","Due Date","Resolution Description","Resolution Action Updated Date","Community Board","X Coordinate (State Plane)","Y Coordinate (State Plane)","Park Borough","Latitude","Longitude","Location"]
new_df.drop(rem1,axis=1,inplace=True)


# In[132]:


new_df.head()


# In[133]:


g=sns.catplot(x='Complaint Type',y='Request_Closing_Time',kind='box',data=new_df)
g.fig.set_figwidth(15)
g.fig.set_figheight(8)
plt.xticks(rotation=90)
plt.ylim((0,2000))


# In[134]:


anova_df=pd.DataFrame()
anova_df['Request_Closing_Time']=new_df['Request_Closing_Time']
anova_df['Complaint']=new_df['Complaint Type']
anova_df.dropna(inplace=True)
anova_df.head()


# In[135]:


lm=ols('Request_Closing_Time~Complaint',data=anova_df).fit()
table=sm.stats.anova_lm(lm)
table


# In[136]:


chi_sq=pd.DataFrame()
chi_sq['Location Type']=new_df['Location Type']
chi_sq['Complaint Type']=new_df['Complaint Type']
chi_sq.dropna(inplace=True)


# In[137]:


data_crosstab=pd.crosstab(chi_sq['Location Type'],chi_sq['Complaint Type'])


# In[138]:


stat,p,dof,expected=chi2_contingency(data_crosstab)
alpha=0.05
if p<=alpha:
    print('Dependent (reject H0)')
else:
    print('Independent (H0 hold true)')


# In[ ]:




