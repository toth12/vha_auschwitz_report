#From here

df_biodata['DateOfBirth'] = pd.to_datetime(df_biodata['DateOfBirth'])
df = df_biodata[['DateOfBirth']]
df = df.sort_values('DateOfBirth').dropna()
df = df.set_index(['DateOfBirth'])
df['value']=range(0,df.shape[0])


df['tvalue'] = df.index
df['delta'] = (df['tvalue']-df['tvalue'].shift()).fillna(0)

container = []
for i,element in enumerate(df['delta'].items()):
    
    if i==0:
        container.append(element[1])

    else:
        duration = element[1]+container[-1]
        container.append(duration)

container = [element.days for element in container]
ser = pd.Series(container)

pp= ser.plot.kde()

pdb.set_trace()

pp.axes[0,1].set_xticks([0,1,2,4.5])

plt.show()
plt.savefig(output_directory+'plots/age_probability.png')

plt.clf()

pdb.set_trace()