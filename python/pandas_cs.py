#Pandas
#Series
s = pd.Series([3, -5, 7, 4], index=['a', 'b', 'c', 'd'])
#DataFrame
data = {'Country': ['Belgium', 'India', 'Brazil'],
 'Capital': ['Brussels', 'New Delhi', 'Brasília'],
 'Population': [11190846, 1303171035, 207847528]}
df = pd.DataFrame(data, columns=['Country', 'Capital', 'Population'])


#Getting
s['b'] #Get one element
 -5
df[1:] #Get subset of a DataFrame
 Country Capital Population
 1 India New Delhi 1303171035
 2 Brazil Brasília 207847528
#Selecting, Boolean Indexing & Setting
#By Position
df.iloc([0],[0]) #Select single value by row &
 'Belgium' column
df.iat([0],[0])
 'Belgium'
 By Label
df.loc([0], ['Country']) #Select single value by row &
 'Belgium' column labels
df.at([0], ['Country']) 'Belgium'
 By Label/Position
df.ix[2] #Select single row of Country Brazil subset of rows Capital Brasília Population 207847528
df.ix[:,'Capital'] #Select a single column of 0 Brussels subset of columns
 1 New Delhi
 2 Brasília df.ix[1,'Capital'] #Select rows and columns
 'New Delhi'
 Boolean Indexing
s[~(s > 1)] #Series s where value is not >1
s[(s < -1) | (s > 2)] s where value is <-1 or >2
df[df['Population']>1200000000] #Use filter to adjust DataFrame
 Setting
s['a'] = 6 Set index a of Series s to 6


#Dropping
s.drop(['a', 'c']) #Drop values from rows (axis=0)
df.drop('Country', axis=1) #Drop values from columns(axis=1)


#Sort & Rank
df.sort_index() #Sort by labels along an axis
df.sort_values(by='Country') #Sort by the values along an axis
df.rank() #Assign ranks to entries

#Retrieving Series/DataFrame Information
#Basic Information
df.shape #(rows,columns)
df.index #Describe index
df.columns #Describe DataFrame columns
df.info() #Info on DataFrame
df.count() #Number of non-NA values
#Summary
df.sum() #Sum of values
df.cumsum() #Cummulative sum of values
df.min()/df.max() #Minimum/maximum values
df.idxmin()/df.idxmax() #Minimum/Maximum index value
df.describe() #Summary statistics
df.mean() #Mean of values
df.median() #Median of values
