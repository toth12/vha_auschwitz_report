# Remove outliers
length_in_minutes = np.array(df_interview_segment_length['length_in_minutes'].values.tolist())

# Reshape it to two dimensional array
length_in_minutes = np.reshape(length_in_minutes, (-1, 1))


clf = IsolationForest(behaviour = 'new', max_samples=12000, random_state = 1, contamination= 'auto')
preds = clf.fit_predict(length_in_minutes)