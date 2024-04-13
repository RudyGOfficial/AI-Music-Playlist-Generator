import re
from collections import Counter

import nltk
import pandas
import matplotlib.pyplot as plt
import pandas as pd
from nltk.stem.porter import PorterStemmer  # stems words in a NL string (ex: for lyrics)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class PlaylistGenerator:
    def __init__(self, song_dataset, customer_single, song_attribute, default_vectorizer=True):
        self.song_dataset_DF = pandas.read_csv(song_dataset)
        self.customer_single_DF = pandas.read_csv(customer_single)
        self.combined_data_frame = None
        self.combineDataFrames()  # Initializes combined_data_frame
        self.song_attribute = song_attribute  # The dataframe column to determine similarity between songs
        self.tokenized_attribute = None
        self.vectorizer = None
        if default_vectorizer:
            self.setVectorizer()
        self.attribute_matrix = None
        self.similarity_matrix = None
        self.similarity_scores = None
        self.song_dictionary = {}

    # Replaces the dataframe with a sampled (i.e: reduced) version of itself
    def sampleDataFrame(self, size=None, drop=True):
        if not size:
            size = int(len(self.song_dataset_DF) / 2)  # Halves dataframe if no size is provided
        self.song_dataset_DF = self.song_dataset_DF.sample(size).reset_index(drop=drop)

    # Drops selected columns from the dataframe and resets the dataframe index
    def dropDataFrameColumns(self, columns_to_drop, axis=1, drop=True):
        # Drops the selected columns from the song dataset & customer single dataframes
        self.song_dataset_DF = self.song_dataset_DF.drop(columns=columns_to_drop, axis=axis).reset_index(
            drop=drop)
        self.customer_single_DF = self.customer_single_DF.drop(columns=columns_to_drop, axis=axis).reset_index(
            drop=drop)

    # Cleans each attribute value to a preferred standard for effective ML processing
    def cleanAttribute(self, lowercase=False, whitespace=False, commaspace=False):
        if lowercase:
            # Convert all characters in each attribute value to lowercase for song dataset & customer single dataframes
            self.song_dataset_DF[self.song_attribute] = self.song_dataset_DF[self.song_attribute].str.lower()
            self.customer_single_DF[self.song_attribute] = self.customer_single_DF[self.song_attribute].str.lower()

        if whitespace:
            # Remove leading word characters followed by whitespace for song dataset & customer single dataframes
            self.song_dataset_DF[self.song_attribute] = self.song_dataset_DF[self.song_attribute].replace(
                r'^\w\s', ' ')
            self.customer_single_DF[self.song_attribute] = self.customer_single_DF[self.song_attribute].replace(
                r'^\w\s', ' ')

        if commaspace:
            # Single space after each comma for all attribute values for song dataset & customer single dataframes
            self.song_dataset_DF[self.song_attribute] = (self.song_dataset_DF[self.song_attribute].
                                                         replace(',', ', ', regex=True).
                                                         replace(r'\s+', ' ', regex=True))
            self.customer_single_DF[self.song_attribute] = (self.customer_single_DF[self.song_attribute].
                                                            replace(',', ', ', regex=True).
                                                            replace(r'\s+', ' ', regex=True))

    # Updates the customer single dataframe and the combined dataframe
    def updateCustomerSingle(self, new_row_data):

        # Convert the new row data dictionary into a dataframe
        self.customer_single_DF = pandas.DataFrame([new_row_data])

        # Update combined_data_frame
        self.combineDataFrames()
        pass

    # Combines the song dataset & customer single dataframes and stores it as another dataframe
    def combineDataFrames(self):
        # Clear the combined data frame if data is present
        if self.combined_data_frame is not None:
            self.combined_data_frame.drop(self.combined_data_frame.index, inplace=True)

        # Concatenate song dataset & customer single dataframes
        combined_data_frame = pandas.concat([self.song_dataset_DF, self.customer_single_DF], ignore_index=True)

        # Assign results
        self.combined_data_frame = combined_data_frame
        return

    # Tokenizes all values within a chosen song attribute
    def setTokenizedAttribute(self):
        self.tokenized_attribute = self.combined_data_frame[self.song_attribute].apply(
            lambda attribute_value: tokenizeValue(attribute_value))
        return self.tokenized_attribute

    # Retrieves the tokenized attribute
    def getTokenizedAttribute(self):
        return self.tokenized_attribute

    # Assigns a vectorizer object to the vectorizer instance variable
    # Analyzes english words & excludes english stop words by default
    def setVectorizer(self, analyzer="word", stop_words="english"):
        self.vectorizer = TfidfVectorizer(analyzer=analyzer, stop_words=stop_words)

    # Fit the vectorizer to a chosen dataframe attribute and transform its values into a TF-IDF feature matrix
    def setAttributeMatrix(self):
        # Builds an attribute matrix with vectorized attribute values
        self.attribute_matrix = self.vectorizer.fit_transform(self.getTokenizedAttribute())
        return self.attribute_matrix

    # Retrieves the attribute matrix
    def getAttributeMatrix(self):
        return self.attribute_matrix

    # Compute the cosine similarity between all pairs of attributes based on their TF-IDF representation
    def setSimilarityMatrix(self):
        self.similarity_matrix = cosine_similarity(self.getAttributeMatrix())
        return self.similarity_matrix

    # Retrieve the similarity matrix
    def getSimilarityMatrix(self):
        return self.similarity_matrix

    # Generate a list of songs from most to least similar with a chosen song as the reference
    def setSongDictionary(self):

        # Clear song dictionary if contents exist
        if self.song_dictionary:
            self.song_dictionary.clear()

        # Extract the track name and artist name from customer_single_DF
        customer_track_name = self.customer_single_DF['track_name'].iloc[0]
        customer_artist_name = self.customer_single_DF['track_artist_name'].iloc[0]

        # Boolean Series indicating whether a given column of each dataframe row matches the customer single
        track_name_series = self.combined_data_frame['track_name'] == customer_track_name
        artist_name_series = self.combined_data_frame['track_artist_name'] == customer_artist_name

        # Combine boolean Series 'track_name' & 'track_artist_name' to better determine song matches
        song_series = track_name_series & artist_name_series  # Series of true & false matches
        if not song_series.any():  # Returns false if no match exists
            print(f"The Song '{customer_track_name}' by '{customer_artist_name}' was not found in the dataset.")
            return False

        # Generate a list of songs
        try:
            # Filter combined_dataframe to only have rows of songs matching customer's single by artist & track name
            true_song_matches = self.combined_data_frame[song_series]

            # Retrieve the index of the first match present in the filtered combined_dataframe
            index = true_song_matches.index[0]  # Get the index of the first row (customer's single)

            # Calculate similarity scores & stores them in descending order
            self.similarity_scores = (
                sorted(list(enumerate(self.similarity_matrix[index])), reverse=True, key=lambda x: x[1]))

            # Retrieve the top similar songs to a chosen track, excluding duplicates
            for song_id in self.similarity_scores:

                # Get a particular column's value of the row matching the current song id
                dataset_track_name = self.combined_data_frame.iloc[song_id[0]]['track_name']  # Get track name
                dataset_artist_name = self.combined_data_frame.iloc[song_id[0]]['track_artist_name']  # Get artist name

                # Build song dictionary
                if (dataset_artist_name, dataset_track_name) == (customer_artist_name, customer_track_name):
                    continue
                elif dataset_artist_name not in self.song_dictionary:
                    self.song_dictionary[dataset_artist_name] = [dataset_track_name]
                elif dataset_track_name not in self.song_dictionary[dataset_artist_name]:
                    self.song_dictionary[dataset_artist_name].append(dataset_track_name)
            return self.similarity_scores, self.song_dictionary
        except IndexError:
            return f"Error: Unable to retrieve recommendations for '{customer_track_name}' by '{customer_artist_name}'."

    # Retrieve the song list
    def getSongDictionary(self):
        return self.song_dictionary

    # Print the song list
    def printSongDictionary(self, quantity=20, first_match_only=True):
        playlist_string = ""
        count = quantity - 1
        for artist, track_list in self.song_dictionary.items():
            if first_match_only:
                playlist_string = playlist_string + f"{quantity - count}. {artist}: {track_list[0]}\n"
                count -= 1
            else:
                for track in track_list:
                    playlist_string = playlist_string + f"{quantity - count}. {artist}: {track}\n"
                    count -= 1
                    if count == -1:
                        break
            if count == -1:
                break
        return playlist_string

    # Opens a window of a particular graph for analysis of the song dataset
    def dataVisualizations(self, *args):
        # Retrieve song dataset dataframe
        df = self.song_dataset_DF

        for option in args:
            if option == 'songsperdecade':

                # Convert album release year column to datetime format
                df['album_release_year'] = pd.to_datetime(df['album_release_year'], format='%Y')

                # Create a new column for decade
                df['decade'] = df['album_release_year'].dt.year // 10 * 10

                # Group by decade and count the number of songs
                songs_per_decade = df.groupby('decade').size()

                # Plotting the bar graph
                songs_per_decade.plot(kind='bar',
                                      title='Number of Songs per Decade',
                                      xlabel='Decade',
                                      ylabel='Number of Songs',
                                      color='skyblue')
                pass
            elif option == 'topartistgenres':

                # Extract all genres and split them into individual words
                all_genres = df['artist_genres'].str.split(',', expand=True).stack().str.strip()

                # Count occurrences of each word
                word_counts = all_genres.value_counts()

                # Select the top 5 most frequent words
                top_words = word_counts.head(5)

                # Plotting the bar graph
                plt.figure(figsize=(10, 6))
                plt.bar(top_words.index, top_words.values, color='skyblue')
                plt.title('Top 5 Most Frequent Artist Sub-Genres')
                plt.xlabel('Genre')
                plt.ylabel('Count')
                pass
            elif option == 'artistgenreskeywords':
                # Combine all genre strings into a single string
                all_genres_str = ', '.join(df['artist_genres'])

                # Extract all English words from the combined string
                english_words = re.findall(r'\b(?:hip hop|new wave|[a-zA-Z]+)\b', all_genres_str)

                # Count occurrences of each English word
                word_counts = Counter(english_words)

                # Get the most common 10 words
                top_words = word_counts.most_common(10)

                # Extract words and counts for plotting
                words, counts = zip(*top_words)

                # Plotting the bar graph
                plt.figure(figsize=(12, 6))
                plt.bar(words, counts, color='skyblue')
                plt.title('Most Common Keywords in Artist Genres')
                plt.xlabel('Keyword')
                plt.ylabel('Count')
                plt.xticks(ha='right')  # Aligns x-axis labels to right of ticks
            else:
                print(f"Invalid option: {option}")
                return

            # Data visualization formatting for improved readability
            plt.xticks(rotation=45)  # Rotate x-axis labels 45 degrees
            plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add y-axis semi-transparent dotted lines
            plt.tight_layout()  # Ensures plot element fit in the display correctly
            plt.show()  # Opens results in a separate window
        return


# Tokenizes a single value within a chosen song attribute
def tokenizeValue(attribute_value, stem=False):

    # Tokenize the input string: convert from a string of text to a list of words
    word_list = nltk.word_tokenize(attribute_value)

    # stem the text input if true, otherwise leave text input not stemmed
    if stem:
        # Initialize an instance of the Porter Stemmer algorithm object from the NLTK library
        stemmer = PorterStemmer()

        # List comprehension stems each token with the stem method and stores the results in a list variable
        stem_list = [stemmer.stem(word) for word in word_list]

        # Join stem list items into a single string with each stem separated by a single whitespace
        token_string = " ".join(stem_list)
    else:
        # Join word list items into a single string with each word separated by a single whitespace
        token_string = " ".join(word_list)
    return token_string
