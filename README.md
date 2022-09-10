# SpotifyRecommender

Demo: https://drive.google.com/file/d/1M7HDwxO94ZaJADHt8ZXExAVMSvyVZR4N/view?usp=sharing

Note: Demo does not show loading time for connecting to Spotify API and generating list of recommendations.

Description:

When a user inputs a playlist URL into the text field on the Home page, the URL is passed to a python script that uses the Spotify API to get features of each track on the playlist (such as genre and tempo, as well as lyrics sentiment analysis using the Textblob library). It then builds a feature vector for each track. 

Using the feature set for the input playlist, the python script compares this to the feature vector for a random set of 500 songs on Spotify, and returns the top 50 songs most similar to the input playlist feature set.

Of these, the top 25 songs are then listed on the webpage using Django, including the track name, artist, album, cosine similarity to the playlist feature set, and a link to the recommended song on Spotify.
