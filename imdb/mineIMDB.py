import pandas as pd
import pdb
import csv

link_df = pd.read_csv('../../../movieratepredictions/links.csv')

movieIds = []
imdbIds  = []

# populate movie id arrays
# do not include ids for which imdb ids do not exist (they don't have data on the movie)
for i in range(link_df.shape[0]):
  if not pd.isnull(link_df.at[i, 'imdbId']):
    movieIds.append(link_df.at[i, 'movieId'])
    imdbIds.append(link_df.at[i, 'imdbId'])

data_df = pd.read_csv('data.tsv', sep='\t')

# ----------BELOW ONLY NEEDS TO BE DONE ONCE----------
# data cleaning
# remove leading useless characters in each entry of this file
# for i in range(data_df.shape[0]):
#   cur = data_df.at[i, 'tconst']
#   cur = cur[2:] # remove 'tt'
#   while(cur[:1] == '0'): # remove leading '0'
#     cur = cur[1:]
#   data_df.at[i, 'tconst'] = cur

# data_df.to_csv('data.tsv', sep='\t', index=False)

# ----------ABOVE ONLY NEEDS TO BE DONE ONCE----------

# for each element in imdbIds, we want to find relevant data in data.tsv file
# then, we want to push corresponding movieId, imdbId, avg. rating, and num rating row

with open('imdbOutput.csv', mode='w+') as toWrite:
  writer = csv.writer(toWrite, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
  writer.writerow(['movieId', 'imdbId', 'avgRating', 'numRatings'])
  for i in range(len(imdbIds)):
    myRow = data_df.loc[data_df['tconst'] == imdbIds[i]]
    if myRow.shape[0] == 1:
      avgRating = myRow.reset_index()['averageRating'][0]
      numVotes = myRow.reset_index()['numVotes'][0]
      # pdb.set_trace()
      # print("movieId: "), print(movieIds[i]), print(" imdbId: "), print(imdbIds[i]), print(
      #     " avgRating: "), print(myRow.reset_index()['averageRating'][0]), print(" numRatings: "), print(myRow.reset_index()['numVotes'][0])
      writer.writerow([movieIds[i], imdbIds[i], avgRating, numVotes])
