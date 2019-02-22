import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import csv
import joblib

# original data files
TRAIN_RATINGS = "../data/movieratepredictions/train_ratings.csv"
TEST_RATINGS = "../data/movieratepredictions/test_ratings.csv"
VAL_RATINGS = "../data/movieratepredictions/val_ratings.csv"

# movie information files
UNFORMATTED_TAG_FILE = "../data/movieratepredictions/genome-scores.csv"
IMDB_FILE = "../imdb/imdbOutput.csv"
TMDB_FILE = "../tmdb/tmdb_ratings.csv"

# intermediate data files
FEATURE_FILE_ALL = "movie_features_all.csv"
FEATURE_FILE_NO_TAGS = "movie_features_no_tags.csv"
FEATURE_FILE_NO_IMDB = "movie_features_no_imdb.csv"
FEATURE_FILE_NO_TMDB = "movie_features_no_tmdb.csv"
FEATURE_FILE_ONLY_TAGS = "movie_features_only_tags.csv"
FEATURE_FILE_ONLY_IMDB = "movie_features_only_imdb.csv"
FEATURE_FILE_ONLY_TMDB = "movie_features_only_tmdb.csv"

# model savenames
SAVENAME_MODEL_ALL = "model_all.joblib"
SAVENAME_MODEL_NO_TAGS = "model_no_tags.joblib"
SAVENAME_MODEL_NO_IMDB = "model_no_imdb.joblib"
SAVENAME_MODEL_NO_TMDB = "model_no_tmdb.joblib"
SAVENAME_MODEL_ONLY_TAGS = "model_only_tags.joblib"
SAVENAME_MODEL_ONLY_IMDB = "model_only_imdb.joblib"
SAVENAME_MODEL_ONLY_TMDB = "model_only_tmdb.joblib"
SAVENAME_OVERALL_AVERAGE = "model_overall_average.joblib"

#test output csv file
OUTPUT_CSV_NAME = "linreg.csv"

# all created files will be in here
DEFAULT_SAVE_DIR = "linear_regression_data"
# shortcut for prepending the default directory to filenames
def filepath(filename):
	return DEFAULT_SAVE_DIR + "/" + filename


PRINT_STEP = 500 # during data formatting, print a status message every PRINT_STEP operations

# there's too much data to store in a numpy array, and we can only train the linear regression alg once. Thus, for training we sample only a random batch of this many datapoints
TRAIN_BATCH_SIZE = 100000

# formats movie information into new CSV files organized so as to be easily extractible into feature vectors
def format_feature_data():
	print("Loading csv's")
	tags_df = pd.read_csv(UNFORMATTED_TAG_FILE)
	imdb_df = pd.read_csv(IMDB_FILE)
	tmdb_df = pd.read_csv(TMDB_FILE)

	# convert tag scores to hashmaps (movie ID -> tag relevance array)
	tagMovieIds = np.unique(list(tags_df['movieId']))
	featureIds = np.unique(list(tags_df['tagId']))
	tags = {}
	for i in range(len(tagMovieIds)):
		temp = tags_df[tags_df['movieId'] == tagMovieIds[i]]
		
		if not np.isnan(np.array(temp)).any():
			tags[tagMovieIds[i]] = list(temp['relevance'])
		
		if i % PRINT_STEP == 0:
			print("Tags: ", i + 1, " / ", len(tagMovieIds))
		
	# convert imdb scores to hashmap (movie ID -> array of score and number of votes)
	imdbMovieIds = np.unique(list(imdb_df['movieId']))
	imdb = {}
	i = 1
	maxVotes = max(list(imdb_df['numRatings']))
	maxRating = 10
	for row in imdb_df.itertuples(index=False):
		try:
			movieId = row.movieId
			avgRating = float(row.avgRating)
			numRatings = float(row.numRatings)
		except:
			continue
        
		if (not (np.isnan(avgRating) or np.isnan(numRatings))):
			imdb[movieId] = [avgRating, numRatings]
        
		if i % PRINT_STEP == 0:
			print("IMDB: ", i, " / ", len(imdb_df))
		i += 1
	
	# convert tmdb scores to hashmap (movie ID -> array of 1 score)
	tmdbMovieIds = np.unique(list(tmdb_df['movieId']))
	tmdb = {}
	i = 1
	maxRating = 10
	for row in tmdb_df.itertuples(index=False):
		try:
			movieId = row.movieId
			rating = float(row.rating)
		except:
			continue
		
		if (not np.isnan(rating)):
			tmdb[movieId] = [rating]
		
		if i % PRINT_STEP == 0:
			print("TMDB: ", i, " / ", len(tmdb_df))
		i += 1

    
	# bring them all together, writing to a file
	with open(filepath(FEATURE_FILE_ALL), mode='w') as write_file_all,\
		open(filepath(FEATURE_FILE_NO_TAGS), mode='w') as write_file_no_tags,\
		open(filepath(FEATURE_FILE_NO_IMDB), mode='w') as write_file_no_imdb,\
		open(filepath(FEATURE_FILE_NO_TMDB), mode='w') as write_file_no_tmdb,\
		open(filepath(FEATURE_FILE_ONLY_TAGS), mode='w') as write_file_only_tags,\
		open(filepath(FEATURE_FILE_ONLY_IMDB), mode='w') as write_file_only_imdb,\
		open(filepath(FEATURE_FILE_ONLY_TMDB), mode='w') as write_file_only_tmdb:
			
		writer_all = csv.writer(write_file_all, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		writer_no_tags = csv.writer(write_file_no_tags, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		writer_no_imdb = csv.writer(write_file_no_imdb, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		writer_no_tmdb = csv.writer(write_file_no_tmdb, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		writer_only_tags = csv.writer(write_file_only_tags, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		writer_only_imdb = csv.writer(write_file_only_imdb, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		writer_only_tmdb = csv.writer(write_file_only_tmdb, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		
		
		# write headers
		headerMain = ["movieId"]
		headerTags = ["tag " + str(fID) for fID in featureIds]
		headerIMDB = ["imdbRating", "imdbVotes"]
		headerTMDB = ["tmdbRating"]
		
		writer_all.writerow(headerMain + headerTags + headerIMDB + headerTMDB)
		writer_no_tags.writerow(headerMain + headerIMDB + headerTMDB)
		writer_no_imdb.writerow(headerMain + headerTags + headerTMDB)
		writer_no_tmdb.writerow(headerMain + headerTags + headerIMDB)
		writer_only_tags.writerow(headerMain + headerTags)
		writer_only_imdb.writerow(headerMain + headerIMDB)
		writer_only_tmdb.writerow(headerMain + headerTMDB)
		
		# find all movie ids referenced
		allMovieIds = np.unique(list(tagMovieIds) + list(imdbMovieIds) + list(tmdbMovieIds))
		
		i = 1
		length = len(allMovieIds)
		for movieId in allMovieIds:
			tag_row = None
			imdb_row = None
			tmdb_row = None
			
			if movieId in tags.keys():
				tag_row = tags[movieId]
			if movieId in imdb.keys():
				imdb_row = imdb[movieId]
			if movieId in tmdb.keys():
				tmdb_row = tmdb[movieId]


			# write to necessary data files
			if tag_row != None:
				writer_only_tags.writerow([movieId] + tag_row)
			if imdb_row != None:
				writer_only_imdb.writerow([movieId] + imdb_row)
			if tmdb_row != None:
				writer_only_tmdb.writerow([movieId] + tmdb_row)
			if tag_row != None and imdb_row != None:
				writer_no_tmdb.writerow([movieId] + tag_row + imdb_row)
			if tag_row != None and tmdb_row != None:
				writer_no_imdb.writerow([movieId] + tag_row + tmdb_row)
			if imdb_row != None and tmdb_row != None:
				writer_no_tags.writerow([movieId] + imdb_row + tmdb_row)
			if tag_row != None and imdb_row != None and tmdb_row != None:
				writer_all.writerow([movieId] + tag_row + imdb_row + tmdb_row)
				
			if i % PRINT_STEP == 0:
				print("Synthesis and writing files: ", i, " / ", length)
			i += 1
        

# loads data from the intermediate data files, loading them into dictionaries
# args: 
# 		filename - file to read from. Allows any FEATURE_FILE_*.csv
# if all args are False, an empty dict will be returned
# returns:
#		dictionary (movieId -> list of floats (features))
def get_feature_dict(filename):
	
	filename = filepath(filename)
	
	# load from file
	features = pd.read_csv(filename)
	
	featureDict = {}
	length = len(features)
	for i in range(length):
		featureInstance = list(features.iloc[i])
		movieId = featureInstance[0]
		featureDict[movieId] = featureInstance[1:] # disinclude the movieID field from the data, everything else is a feature
		
		if i % PRINT_STEP == 0:
			print("Loading from ", filename, ": ", i, " / ", length)
	
	return featureDict


# returns a numpy array of ratings values gleaned from the ratings pandas dataset
# args:
#		ratings: pandas dataset loaded from TRAIN_RATINGS, VAL_RATINGS, or TEST_RATINGS
#		feature_dict: any dictionary of (movieId -> list of floats)
# returns:
#		x: numpy array, dimensions (N x D), where N <= len(ratings) (accounting for instances where the feature dictionary has no entry for a movieId) and D = len(feature_dict[x])
#		y: numpy array, dimensions (N x 1), same N value as the x array
# if ratings is from TEST_RATINGS, which has no 'ratings' column, an empty numpy array will be returned
def getTrainXY(ratings, feature_dict):
	x = []
	y = []
	
	# check if this is TEST_RATINGS
	containsLabels = ('rating' in list(ratings))
	
	# get random batch of size TRAIN_BATCH_SIZE (only if there's more data than TRAIN_BATCH_SIZE)
	indexes = np.random.choice(np.arange(len(ratings)), min(TRAIN_BATCH_SIZE, len(ratings)))
	
	length = len(indexes)
	for i in range(len(indexes)):
		idx = indexes[i]
		row = ratings.iloc[idx]
		
		movieId = row['movieId']
		if movieId in feature_dict.keys():
			x.append(feature_dict[movieId])
			
			if containsLabels:
				y.append([row['rating']])
		
		if i % PRINT_STEP == 0:
			print("Extracting x-y values: ", i, " / ", length)
		i += 1
		
	print("Converting huge lists to numpy arrays...")
	y = np.array(y)
	x = np.array(x)
	return x, y


# Linear Regression Class
class LinRegModel:
	def setup(self):
		# load feature dicts
		self.loadFeatureDicts()
		
		# use feature dictionaries to convert training data into usable form
		unformatted_training_data = pd.read_csv(TRAIN_RATINGS)
		
		# train a model for each subset of data
		print("Model All: Getting data...")
		x, y = getTrainXY(unformatted_training_data, self.features_all)
		print("Model All: Training...")
		self.model_all = self.train(x, y)
		
		print("Model No Tags: Getting data...")
		x, y = getTrainXY(unformatted_training_data, self.features_no_tags)
		print("Model No Tags: Training...")
		self.model_no_tags = self.train(x, y)
		
		print("Model No IMDB: Getting data...")
		x, y = getTrainXY(unformatted_training_data, self.features_no_imdb)
		print("Model No IMDB: Training...")
		self.model_no_imdb = self.train(x, y)
		
		print("Model No TMDB: Getting data...")
		x, y = getTrainXY(unformatted_training_data, self.features_no_tmdb)
		print("Model NO TMDB: Training...")
		self.model_no_tmdb = self.train(x, y)
		
		print("Model Only Tags: Getting data...")
		x, y = getTrainXY(unformatted_training_data, self.features_only_tags)
		print("Model Only Tags: Training...")
		self.model_only_tags = self.train(x, y)
		
		print("Model Only IMDB: Getting data...")
		x, y = getTrainXY(unformatted_training_data, self.features_only_imdb)
		print("Model Only IMDB: Training...")
		self.model_only_imdb = self.train(x, y)
		
		print("Model Only TMDB: Getting data...")
		x, y = getTrainXY(unformatted_training_data, self.features_only_tmdb)
		print("Model Only TMDB: Training...")
		self.model_only_tmdb = self.train(x, y)
		
		# find overall average rating, for use when we have no stored data on a movie ID
		print("Finding overall average...")
		self.overallAverage = np.mean(list(unformatted_training_data['rating']))
		
	
	# pretty obvious - returns a trained model
	# args:
	#		x: numpy array of floats (N x D)
	#		y: numpy array of floats (N x 1)
	def train(self, x, y):
		return LinearRegression().fit(x, y)
		
	
	# predicts using given models
	# args:
	#		x: list of movieIds (int >= 1)
	# returns:
	#		y: list of floats / predicted ratings
	def predict(self, movieIdList):
		i = 1
		length = len(movieIdList)
		y = []
		for movieId in movieIdList:
			y.append(self.predict_one(movieId))
			
			if i % PRINT_STEP == 0:
				print("Predicting: ", i, " / ", length)
			i += 1

		return y
			
	
	# predicts one case
	# args:
	#		movieId: ID for one movie
	# returns:
	#		float for predicted rating
	def predict_one(self, movieId):
		# check in all
		if movieId in self.features_all.keys():
			x = np.array(self.features_all[movieId])
			x = x.reshape(1, -1)
			return self.model_all.predict(x)[0][0]
		
		elif movieId in self.features_no_tags.keys():
			x = np.array(self.features_no_tags[movieId])
			x = x.reshape(1, -1)
			return self.model_no_tags.predict(x)[0][0]
		
		elif movieId in self.features_no_imdb.keys():
			x = np.array(self.features_no_imdb[movieId])
			x = x.reshape(1, -1)
			return self.model_no_imdb.predict(x)[0][0]
			
		elif movieId in self.features_no_tmdb.keys():
			x = np.array(self.features_no_tmdb[movieId])
			x = x.reshape(1, -1)
			return self.model_no_tmdb.predict(x)[0][0]
			
		elif movieId in self.features_only_tags.keys():
			x = np.array(self.features_only_tags[movieId])
			x = x.reshape(1, -1)
			return self.model_only_tags.predict(x)[0][0]
		
		elif movieId in self.features_only_imdb.keys():
			x = np.array(self.features_only_imdb[movieId])
			x = x.reshape(1, -1)
			return self.model_only_imdb.predict(x)[0][0]
			
		elif movieId in self.features_only_tmdb.keys():
			x = np.array(self.features_only_tmdb[movieId])
			x = x.reshape(1, -1)
			return self.model_only_tmdb.predict(x)[0][0]
			
		else:
			return self.overallAverage
		
	
	# use on VAL_RATINGS or TRAIN_RATINGS - REQUIRES a 'rating' column
	# returns errorMeasure of results
	def validate(self, ratings_filename):
		print("Reading csv file...")
		ratings = pd.read_csv(ratings_filename)
		
		y_real = list(ratings['rating'])
		movieIdList = list(ratings['movieId'])
		
		print("Predicting...")
		y_pred = self.predict(movieIdList)
		
		return self.errorMeasure(y_real, y_pred)
		
		
	# writes a prediction ratings list to a csv file (OUTPUT_CSV_NAME)
	def test(self):
		print("Reading test ratings file...")
		ratings = pd.read_csv(TEST_RATINGS)
		
		movieIdList = list(ratings['movieId'])
		
		print("Predicting...")
		y_pred = self.predict(movieIdList)
		
		print("Writing to file...")
		with open(filepath(OUTPUT_CSV_NAME), mode='w') as write_file:
			writer = csv.writer(write_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
			
			# headers
			writer.writerow(["Id", "rating"])
			
			length = len(movieIdList)
			for i in range(length):
				writer.writerow([i, y_pred[i]])
				
				if i % PRINT_STEP == 0:
					print("Writing output: ", i, " / ", length)
		
			
		
	# Returns RMSE of given arrays
	def errorMeasure(self, y_real, y_pred):
		# make sure arrrays are correctly typed
		y_real = np.array(y_real)
		y_pred = np.array(y_pred)
		
		return np.sqrt(((y_pred - y_real) ** 2).mean())
		
		
	
	# saves the internal models to the default directory
	def save(self):
		# save learnt models
		joblib.dump(self.model_all, filepath(SAVENAME_MODEL_ALL))
		joblib.dump(self.model_no_tags, filepath(SAVENAME_MODEL_NO_TAGS))
		joblib.dump(self.model_no_imdb, filepath(SAVENAME_MODEL_NO_IMDB))
		joblib.dump(self.model_no_tmdb, filepath(SAVENAME_MODEL_NO_TMDB))
		joblib.dump(self.model_only_tags, filepath(SAVENAME_MODEL_ONLY_TAGS))
		joblib.dump(self.model_only_imdb, filepath(SAVENAME_MODEL_ONLY_IMDB))
		joblib.dump(self.model_only_tmdb, filepath(SAVENAME_MODEL_ONLY_TMDB))
		
		# save average
		joblib.dump(self.overallAverage, filepath(SAVENAME_OVERALL_AVERAGE))
		
		
	# loads the internal models from the default directory
	def load(self):
		# load learnt models
		self.model_all = joblib.load(filepath(SAVENAME_MODEL_ALL))
		self.model_no_tags = joblib.load(filepath(SAVENAME_MODEL_NO_TAGS))
		self.model_no_imdb = joblib.load(filepath(SAVENAME_MODEL_NO_IMDB))
		self.model_no_tmdb = joblib.load(filepath(SAVENAME_MODEL_NO_TMDB))
		self.model_only_tags = joblib.load(filepath(SAVENAME_MODEL_ONLY_TAGS))
		self.model_only_imdb = joblib.load(filepath(SAVENAME_MODEL_ONLY_IMDB))
		self.model_only_tmdb = joblib.load(filepath(SAVENAME_MODEL_ONLY_TMDB))
		
		# load average
		self.overallAverage = joblib.load(filepath(SAVENAME_OVERALL_AVERAGE))

		# load feature dictionaries
		self.loadFeatureDicts()
		
	def loadFeatureDicts(self):
		print("Loading movie-to-feature dictionaries from intermediate data files")
		self.features_all = get_feature_dict(FEATURE_FILE_ALL)
		self.features_no_tags = get_feature_dict(FEATURE_FILE_NO_TAGS)
		self.features_no_imdb = get_feature_dict(FEATURE_FILE_NO_IMDB)
		self.features_no_tmdb = get_feature_dict(FEATURE_FILE_NO_TMDB)
		self.features_only_tags = get_feature_dict(FEATURE_FILE_ONLY_TAGS)
		self.features_only_imdb = get_feature_dict(FEATURE_FILE_ONLY_IMDB)
		self.features_only_tmdb = get_feature_dict(FEATURE_FILE_ONLY_TMDB)
		

if __name__ == "__main__":
	format_feature_data()
	
	model = LinRegModel()
	
	# either setup from scratch, or load model data
	model.setup()
	#model.load()
	
	# save data
	model.save()
	
	print("RMSE: ", model.validate(VAL_RATINGS)[0][0])
	
	#model.test()
