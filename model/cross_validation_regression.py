import pandas as pd
import numpy as np
import random
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt; plt.rcdefaults()
from scipy import stats
import matplotlib.ticker as mtick

file1 = 'house_no_missing.csv'
file2 ='house_with_missing.csv'

#brief function describes data  

def brief(fName):
    data=pd.read_csv(fName)
    numCols= list(data.select_dtypes(include=[np.number]).columns.values)
    symCols= list(data.select_dtypes(exclude=[np.number]).columns.values)
    
    #Data frame for Real Attributes
    realStats=pd.DataFrame(
            {'Attribute_Name':list(data.select_dtypes(include=[np.number]).columns.values),
             'Attribute_ID':[data.columns.get_loc(c)+1 for c in data.columns if c in numCols],
             'Missing':data.select_dtypes(include=[np.number]).isnull().sum(),#summing null counts
             'Mean': data.select_dtypes(include=[np.number]).mean().round(2),
             'Median':data.select_dtypes(include=[np.number]).median().round(2),
             'Sdev':data.select_dtypes(include=[np.number]).std().round(2),
             'Min':data.select_dtypes(include=[np.number]).min().round(2),
             'Max':data.select_dtypes(include=[np.number]).max().round(2)
                    })
    realStats=realStats.reset_index(drop=True)
    realStats.index+=1 #starting index from 1
   
    MCVs= data.select_dtypes(exclude=[np.number]).apply(pd.value_counts).head(3) #counts top 3 values
    
    #Data Frame for symbolic attributes
    symStats=pd.DataFrame(
            {'Attribute_ID':[data.columns.get_loc(c)+1 for c in data.columns if c in symCols],
            'Attribute_Name':list(data.select_dtypes(exclude=[np.number]).columns.values),
            'Missing':data.select_dtypes(exclude=[np.number]).isnull().sum(),
            'arity': data.select_dtypes(exclude=[np.number]).nunique(), #number of unique values
            'MCVs_counts':"".join(str(i) + " ("+str(row[0])+") " for i, row in MCVs.iterrows())
            })
    symStats=symStats.reset_index(drop=True)
    symStats.index+=1
    
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'brief function output for '+ fName
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'This dataset has '+str(len(data))+ ' Rows '+str(len(data.columns)) + ' Attributes'
    print ' '
    print 'real valued attributes'
    print '----------------------'
    print realStats[['Attribute_ID','Attribute_Name','Missing','Mean','Median','Sdev','Min','Max']]
    print 'symbolic attributes'
    print '----------------------'
    print symStats[['Attribute_ID','Attribute_Name','Missing','arity','MCVs_counts']]

#running brief function on both datasets

brief(file1)
brief(file2)


#defining import function to bring data in desired format for remaining analysis
def import_data(fName):
    df=pd.read_csv(file1)
    n,m = df.shape 
    colnames = df.columns.tolist()    
    if colnames.index('house_value')!=m-1:
        colnames.append(colnames.pop(colnames.index('house_value')))        
    df=df[colnames]
    df=df[colnames].fillna(df[colnames].median()) #replacing missing values with median
    
    return df

#load house_no_missing.csv
df1=import_data(file1)
df2=import_data(file2)


#Density Plots of each variable
df1.plot(kind='density',title='Density Plot', subplots=True, layout=(4,2),sharex=False, 
         legend=False,fontsize=7, figsize=(10,8))
[ax.legend(loc=1, fontsize=8) for ax in plt.gcf().axes]
plt.show()

#Correlation Matrix
colnames=df1.select_dtypes(include=[np.number]).columns.tolist()
colnames = [c.replace('_',' ').title() for c in colnames]

correlations = df1.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,8,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(colnames,rotation=90)
ax.set_yticklabels(colnames)
ax.grid(False)
plt.show()


#Scatter Plot: House value vs crime rate and house value vs log(crime rate)
plt.subplot(211)
plt.scatter(df1['Crime_Rate'], df1['house_value'],alpha=0.4)
plt.title('Scatter Plot of Crime Rate and House Value')
plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e')) #formatting ylabels in exponential
plt.xlabel('Crime Rate')
plt.ylabel('House Value')
    
plt.subplot(212)
plt.scatter(np.log(df1['Crime_Rate']), df1['house_value'],alpha=0.4) #log transformation of crime rate
plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))
plt.xlabel('log(Crime Rate)')
plt.ylabel('House Value')
plt.show()


# Connect-the-dots model that learns from train set and is being tested using test set
# Assumes inputs are pandas data frames
# Assumes the last column of data is the output dimension
    
def get_pred_dots(train,test):
    n,m = train.shape # number of rows and columns
    X = train.iloc[:,:m-1]# get training input data
    query = test.iloc[:,:m-1]# get test input data
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='kd_tree').fit(X)
    distances, nn_index = nbrs.kneighbors(query)# Get two nearest neighbors
    pred = (train.iloc[nn_index[:,0],m-1].values+train.iloc[nn_index[:,1],m-1].values)/2.0
    return pred

# Linear model
# Assumes the last column of data is the output dimension
def get_pred_lr(train,test):
    n,m = train.shape # number of rows and columns
    x_train=train.iloc[:,:m-1] #assumes last columns as output variable
    y_train=train.iloc[:,-1] #predictor variable of training set
    x_test=test.iloc[:,:m-1] #testing set
    regressor = LinearRegression()  
    regressor.fit(x_train, y_train)  
    y_pred = regressor.predict(x_test)
    return y_pred #returns predicted value of test set

# Default predictor model
# Assumes the last column of data is the output dimension
def get_pred_default(train,test):
    n_train,m_train = train.shape
    n_test,m_test=test.shape
    pred = [round(train.iloc[:,-1].mean(),2)]*n_test #predicts output to be mean of test set
    return pred
	

#function to perform k-fold cross validation	
def do_cv(df, output, k, model):
    n,m=df.shape
    colnames = df.columns.tolist()
    if colnames.index(output)!=m-1:
        colnames.append(colnames.pop(colnames.index(output))) #shifting output column to the end
    df=df[colnames]
    rand_index=random.sample(range(n), n) #taking random sample indexes
    chunks=[rand_index[i::k] for i in xrange(k)] #dividing random indexes into k chunks
    score=['NaN']*len(chunks) #initialising score to be returned
    for i in range(0,len(chunks)):
        test_index=chunks[i]    #one chunk becomes test
        train_index=list(set(rand_index) - set(test_index)) #remaining chunks become train
        test=df.iloc[test_index,:]
        train=df.iloc[train_index,:]
        prediction=model(train.select_dtypes(include=[np.number]), #excluding symbolic attributes
                         test.select_dtypes(include=[np.number]))   #excluding symbolic attributes
        true_values=test.iloc[:,-1].tolist()
        #calculating mean squared error using predicted and true values
        mse= round((1/float(len(prediction)))*(sum((np.array(prediction)-np.array(true_values))**2)))
        score[i]=mse #adding the MSE of each chunk into the score list
    return score


#preparing dataframe for input to model taking crime rate as the independent variable
df_model=pd.DataFrame(
        {'Crime_Rate':np.log(df1['Crime_Rate']), #log transformation of crime rate
         'house_value':df1['house_value']}
        )    

#computing scores from each model for k=10 (not required but useful to test)
dots_score = do_cv(df_model, 'house_value',10,get_pred_dots)        
lr_score = do_cv(df_model, 'house_value',10,get_pred_lr)
def_score = do_cv(df_model, 'house_value',10,get_pred_default)

#plotting bar graph of MSE obtained for each k by each of the three models
def model_plots(k, dots_score, lr_score, def_score): 
    fig,ax = plt.subplots()
    objects = [i+1 for i in range(0,k)]
    x_pos = np.arange(len(objects))
    axes = plt.gca()
    axes.set_ylim([0,15000000000])
    plt.bar(x_pos, dots_score, color='darkblue', alpha=0.3)
    plt.xlabel('Fold No')
    plt.ylabel('Score')
    plt.title('Connect-the-Dots Model')
    plt.xticks(x_pos, objects)
    plt.axhline(y=np.mean(dots_score), color='k', linestyle='dashed', linewidth=1)
    plt.show()
    
    fig,ax = plt.subplots()
    objects = [i+1 for i in range(0,k)]
    x_pos = np.arange(len(objects))
    axes = plt.gca()
    axes.set_ylim([0,15000000000])
    plt.bar(x_pos, lr_score,color='darkblue', alpha=0.8)
    plt.xlabel('Fold No')
    plt.ylabel('Score')
    plt.title('Linear Model')
    plt.xticks(x_pos, objects)
    plt.axhline(y=np.mean(lr_score), color='k', linestyle='dashed', linewidth=1)
    plt.show()
    
    fig,ax = plt.subplots()
    objects = [i+1 for i in range(0,k)]
    x_pos = np.arange(len(objects))
    axes = plt.gca()
    axes.set_ylim([0,15000000000])
    plt.bar(x_pos, def_score, color='darkblue', alpha=0.5)
    plt.xlabel('Fold Num')
    plt.ylabel('Score')
    plt.title('Default Model')
    plt.xticks(x_pos, objects)
    plt.axhline(y=np.mean(def_score), color='k', linestyle='dashed', linewidth=1)
    plt.show()


model_plots(10, dots_score, lr_score, def_score) #Show the three bar graphs for K=10

#LEAVE ONE OUT IMPLEMENTATION
#----------------------------------------------------------------------#
k=len(df_model) 

#compute MSE scores
dots_score = do_cv(df_model, 'house_value',k,get_pred_dots)        
lr_score = do_cv(df_model, 'house_value',k,get_pred_lr)
def_score = do_cv(df_model, 'house_value',k,get_pred_default)

model_names=['Linear','Default Predictor','Connect-The-Dots']
model_means=[np.mean(lr_score),np.mean(def_score),np.mean(dots_score)] #calculate mean of MSE of each model
model_ci=[stats.t.interval(0.95, len(lr_score), loc=np.mean(lr_score), scale=stats.sem(lr_score)),
stats.t.interval(0.95, len(def_score), loc=np.mean(def_score), scale=stats.sem(def_score)),
stats.t.interval(0.95, len(dots_score), loc=np.mean(dots_score), scale=stats.sem(dots_score))] #calculating confidence interval

#data frame to compare the three models
comparison = pd.DataFrame({
        'Model_Name':model_names,
        'Mean':model_means,
        'Conf_Interval':model_ci})


import matplotlib.pyplot as plt; plt.rcdefaults()
plt.style.use('ggplot')
# get the range of the confidence interval for each model
y_r = [model_means[i] - model_ci[i][1] for i in range(len(model_ci))]

#plotting the mean of MSE and confidence interval of each model
plt.bar(range(len(model_means)), model_means, yerr=y_r, color='PURPLE',alpha=0.7, align='center')
plt.xticks(range(len(model_means)), model_names)
plt.xlabel('Model')
plt.ylabel('Score')
plt.title('Leave-one-out Model Comparison')
plt.show()

print comparison
