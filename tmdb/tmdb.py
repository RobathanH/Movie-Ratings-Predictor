# this is just a playground, testing things related to tmdb here
import os
import csv
import sys
import requests
import tmdbsimple as tmdb
tmdb.API_KEY = 'f13f38f9a3ca23dee3b21619aad7a0c5'

with open(sys.argv[1], mode='r') as csv_file, open('./tmdb.csv', mode='w+') as write_file:
  csv_reader = csv.DictReader(csv_file)
  writer = csv.writer(write_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
  writer.writerow(['movieId', 'rating'])
  line_count = 0
  for row in csv_reader:
    if line_count % 1000 == 0:
      write_file.flush()
    if line_count == 0:
      line_count += 1

    movie_id = row['movieId'] # get generic movieId
    tmdbId = row['tmdbId'] # get associated tmdbId
    if (tmdbId.isnumeric()):
      movie = tmdb.Movies(tmdbId)
      try:
        res = movie.info()
      except requests.exceptions.HTTPError as err:
        writer.writerow([movie_id, None])
        line_count += 1
        continue
      scaled_rating = float(movie.vote_average/2)
      writer.writerow([movie_id, scaled_rating])
    else:
      writer.writerow([movie_id, None])
    line_count += 1
