from matplotlib import artist
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from bs4 import BeautifulSoup
import requests

import pandas as pd
import numpy as np
import math
import random
import os

from spotifyrec.models import SongRec
from requests.exceptions import ReadTimeout


# Extract information about each track in a playlist
def getPlaylistInfoDf(tracks):
    # List of dictionaries containing info on each track
    trackInfoDicts = []

    for track in tracks:
        # Track name, URI, and popularity
        track_name = track["track"]["name"]
        track_URI = track["track"]["uri"]
        track_pop = track["track"]["popularity"]

        # Main artist
        artist_URI = track["track"]["artists"][0]["uri"]
        artist_info = sp.artist(artist_URI)
        artist_name = track["track"]["artists"][0]["name"]
        artist_pop = artist_info["popularity"]
        artist_genres = artist_info["genres"]

        # Album info
        album_name = track["track"]["album"]["name"]
        album_URI = track["track"]["album"]["uri"]
        album_artist_names = [artist["name"] for artist in track["track"]["album"]["artists"]]
        album_artist_URIs = [artist["uri"] for artist in track["track"]["album"]["artists"]]
        album_artist_genres = list(set([genre for artistURI in album_artist_URIs for genre in sp.artist(artistURI)["genres"]]))# if genre not in album_artist_genres]

        # Put all track info in dictionary
        trackDict = {
            "track_name": track_name, 
            "track_URI": track_URI,
            "track_pop": track_pop,
            "artist_name": artist_name, 
            "artist_URI": artist_URI,
            "artist_pop": artist_pop,
            "album_name": album_name,
            "album_URI": album_URI,
            "album_artist_names": album_artist_names,
            "album_artist_URIs": album_artist_URIs,
            "album_artist_genres": ["_".join(genre.split(" ")) for genre in album_artist_genres]
        }
        
        # Add info about audio features
        track_audio_features = sp.audio_features(track_URI)[0]
        trackDict.update(track_audio_features)

        # Add dictionary to list
        trackInfoDicts.append(trackDict)

    trackInfoDf = pd.DataFrame(trackInfoDicts)
    return trackInfoDf

# Extract information about all songs
def getAllSongsInfoDf(allSongsDf):
    # List of dictionaries containing info on each track
    trackInfoDicts = []

    for index, row in allSongsDf.iterrows():
        if pd.isnull(row["album_uri"]):
            album_artist_names = []
            album_artist_URIs = []
            album_artist_genres = []
        else:
            album_artist_names = [artist["name"] for artist in sp.album(row["album_uri"])["artists"]]
            album_artist_URIs = [artist["uri"] for artist in sp.album(row["album_uri"])["artists"]]
            album_artist_genres = list(set([genre for artistURI in album_artist_URIs for genre in sp.artist(artistURI)["genres"]]))# if genre not in album_artist_genres]

        # Put all track info in dictionary
        trackDict = {
            "track_name": row["track_name"], 
            "track_URI": row["track_uri"],
            "track_pop": sp.track(row["track_uri"])["popularity"],
            "artist_name": row["artist_name"], 
            "artist_URI": row["artist_uri"],
            "artist_pop": sp.artist(row["artist_uri"])["popularity"],
            "album_name": row["album_name"],
            "album_URI": row["album_uri"],
            "album_artist_names": album_artist_names,
            "album_artist_URIs": album_artist_URIs,
            "album_artist_genres": ["_".join(genre.split(" ")) for genre in album_artist_genres]
        }
        # Add info about audio features
        track_audio_features = sp.audio_features(row["track_uri"])[0]
        trackDict.update(track_audio_features)
        trackInfoDicts.append(trackDict)

    trackInfoDf = pd.DataFrame(trackInfoDicts)
    return trackInfoDf


# Incorporate lyrics 
# scrape lyrics from genius
def get_lyrics(artistnames, songname):
    artistnameURL = str("-".join("-and-".join(artistnames).strip().split(" ")))
    songnameURL = str(songname.replace(' ','-')) if ' ' in songname else str(songname)
    fullURL = 'https://genius.com/' + artistnameURL + '-' + songnameURL + '-' + 'lyrics'
    page = requests.get(fullURL)
    html = BeautifulSoup(page.text, 'html.parser')
    lyrics1 = html.find("div", class_="lyrics")
    lyrics2 = html.find("div", class_="Lyrics__Container-sc-1ynbvzw-6 YYrds")
    if lyrics1:
        lyrics = lyrics1.get_text(strip=True, separator="; ")
    elif lyrics2:
        lyrics = lyrics2.get_text(strip=True, separator="; ")
    elif lyrics1 == lyrics2 == None:
        lyrics = None

    return lyrics

# Add lyrics to df
def addLyricsData(trackInfoDf):
    for index, row in trackInfoDf.iterrows():
        trackLyrics = get_lyrics(row["album_artist_names"], row["track_name"])
        trackInfoDf.loc[index, "lyrics"] = trackLyrics

    # Lyrics Analysis
    mean_subj = trackInfoDf["lyrics"].dropna().apply(lambda text: TextBlob(text).sentiment[1]).mean()
    mean_pol = trackInfoDf["lyrics"].dropna().apply(lambda text: TextBlob(text).sentiment[0]).mean()
    trackInfoDf["lyrics_subjectivity"] = trackInfoDf["lyrics"].apply(lambda text: mean_subj if (text == None or (not isinstance(text, str)) or len(text) == 0) else TextBlob(text).sentiment[1])
    trackInfoDf["lyrics_polarity"] = trackInfoDf["lyrics"].apply(lambda text: mean_pol if (text == None or (not isinstance(text, str)) or len(text) == 0) else TextBlob(text).sentiment[0])

    # Add columns for text categories for subjectivity and polarity
    subjectivity_categories = ["Low", "Medium", "High"]
    polarity_categories = ["Negative", "Neutral", "Positive"]

    trackInfoDf["lyrics_subjectivity_cat"] = trackInfoDf["lyrics_subjectivity"].apply(lambda subj: subjectivity_categories[min(2, max(0, math.floor(subj * 3)))])
    trackInfoDf["lyrics_polarity_cat"] = trackInfoDf["lyrics_polarity"].apply(lambda pol: polarity_categories[min(2, max(0, math.floor(pol * 3)))])
    
    return trackInfoDf 


# One-hot encodings
def one_hot_encodings(df, col, colAbbrev):
    tf_df = pd.get_dummies(df[col])
    features = tf_df.columns
    tf_df.columns = [colAbbrev + "|" + str(i) for i in features]
    tf_df.reset_index(drop=True, inplace=True)
    return tf_df


# Create feature set
def create_feature_set(df, float_cols):
    # Term Frequency-Inverse Document Frequency on list of genres in training playlist
    # to prevent over-weighting of less common genres
    tfidf = TfidfVectorizer()
    tfidf_matrix =  tfidf.fit_transform(df['album_artist_genres'].apply(lambda row: " ".join(row)))
    genre_df = pd.DataFrame(tfidf_matrix.toarray())
    genre_df.columns = ['genre' + "|" + i for i in tfidf.get_feature_names()]
    # genre_df.drop(columns='genre|unknown') # drop unknown genre
    genre_df.reset_index(drop = True, inplace=True)

    # One-hot encodings for subjectivity, polarity, key, and mode
    subject_ohe = one_hot_encodings(df, 'lyrics_subjectivity_cat','subject') * 0.3
    polar_ohe = one_hot_encodings(df, 'lyrics_polarity_cat','polar') * 0.5
    key_ohe = one_hot_encodings(df, 'key','key') * 0.5
    mode_ohe = one_hot_encodings(df, 'mode','mode') * 0.5
    

    # Normalize popularity columns
    scaler = MinMaxScaler()
    pop = df[["artist_pop","track_pop"]].reset_index(drop = True)
    pop_scaled = pd.DataFrame(scaler.fit_transform(pop), columns = pop.columns) * 0.2 
    
    # Normalize audio columns
    scaler = MinMaxScaler()
    floats = df[float_cols].reset_index(drop = True)
    floats_scaled = pd.DataFrame(scaler.fit_transform(floats), columns = floats.columns) * 0.2
    
    # Concanenate all features
    feature_set = pd.concat([genre_df, floats_scaled, pop_scaled, subject_ohe, polar_ohe, key_ohe, mode_ohe], axis = 1)
    
    # Add track name and id
    feature_set['id'] = df['id'].values

    return feature_set


def generate_recommendations(allSongsInfoDf, playlist_vector, no_overlap_complete_feature_set):
    no_overlap_df = allSongsInfoDf[allSongsInfoDf['id'].isin(no_overlap_complete_feature_set['id'].values)]

    # Find cosine similarity between the playlist and the complete song set
    no_overlap_df['cos_sim'] = cosine_similarity(no_overlap_complete_feature_set.drop('id', axis=1).values, 
                                playlist_vector.values.reshape(1, -1))[:,0]
    top_50_recs_df = no_overlap_df.sort_values('cos_sim', ascending = False).head(50)
    
    return top_50_recs_df


def init(playlist_link):
    # Config variables: client_id, client_secret
    print("Input Playlist Link:")
    print(playlist_link)
    
    client_id = os.environ['client_id']
    client_secret = os.environ['client_secret']

    # Authenticate without signing into an account
    client_credentials_manager = SpotifyClientCredentials(
        client_id=client_id,
        client_secret=client_secret)

    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager, requests_timeout=10, retries=2)

    playlist_URI = playlist_link.split("/")[-1].split("?")[0]

    # Select 500 random rows (songs) from the first 5,000 rows in the full dataset (>16,000 total)
    allSongDf = pd.read_csv("https://raw.githubusercontent.com/enjuichang/PracticalDataScience-ENCA/main/data/raw_data.csv", 
                            index_col=False, 
                            skiprows=(lambda rIndex: rIndex != 0 and rIndex not in random.sample(range(5000), 500)))


    # Get track, artist, and album info for tracks in training playlist
    trainingTracks = sp.playlist_tracks(playlist_URI)["items"]
    trainingInfoDf = getPlaylistInfoDf(trainingTracks)

    # Add lyrics data to training tracks
    trainingInfoDf = addLyricsData(trainingInfoDf)

    # Add lyrics data to all song tracks
    allSongsInfoDf = getAllSongsInfoDf(allSongDf)
    allSongsInfoDf = addLyricsData(allSongsInfoDf)

    # Narrow down columns to only features relevant for recommendation prediction
    featureCols = ["track_pop", "artist_pop", 
                    "album_artist_genres", "danceability", "energy", "key", "loudness", "mode",
                    "speechiness", "acousticness", "instrumentalness", "liveness", "valence",
                    "tempo", "time_signature", "lyrics_subjectivity", "lyrics_polarity", 
                    "lyrics_subjectivity_cat", "lyrics_polarity_cat", "id", "track_name"]
    trainDf = trainingInfoDf[[col for col in featureCols]]

    allSongsTestDf = allSongsInfoDf[[col for col in featureCols]]

    # Add rows in trainDf to allSongsTestDf
    allSongsTestDf = pd.concat([allSongsTestDf, trainDf])


    # Import all song data (about other songs not in training playlist)

    # Generate features
    training_float_cols = trainDf.dtypes[trainDf.dtypes == 'float64'].index.values
    all_song_float_cols = allSongsTestDf.dtypes[allSongsTestDf.dtypes == 'float64'].index.values

    complete_feature_set = create_feature_set(allSongsTestDf, float_cols=all_song_float_cols)
    training_feature_set = complete_feature_set[complete_feature_set['id'].isin(trainDf['id'].values)]
    no_overlap_complete_feature_set = complete_feature_set[~complete_feature_set['id'].isin(trainDf['id'].values)]

    playlist_vector = training_feature_set.drop(['id'], axis=1).sum(axis=0)

    top_50_recs_df = generate_recommendations(allSongsInfoDf, playlist_vector, no_overlap_complete_feature_set)

    for index, row in top_50_recs_df.iterrows():
        newObj = SongRec(
            track=row["track_name"],
            trackURI=row["track_URI"][row["track_URI"].index("track:") + 6 :], # get only the part in the URL
            artist=row["artist_name"],
            album=row["album_name"],
            similarity=row["cos_sim"] # float
            )
        newObj.save()
        print(newObj)
