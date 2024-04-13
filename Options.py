import sklearn
import nltk
import pandas
import matplotlib


# Prints the Python Library Versions used in this project
def printLibraryVersions():
    print('The scikit-learn version is {}.'.format(sklearn.__version__))
    print('The nltk version is {}.'.format(nltk.__version__))
    print('The pandas version is {}.'.format(pandas.__version__))
    print('The matplotlib version is {}.'.format(matplotlib.__version__))


# Customize pandas print display
def customizePrintDisplay(max_columns=None, width=None, max_colwidth=None):
    pandas.set_option('display.max_columns', max_columns)  # Sets to print all columns (no truncation)
    pandas.set_option('display.width', width)  # Sets to print all columns in a single line (no wrap)
    pandas.set_option('display.max_colwidth', max_colwidth)  # Sets to print all data for each cell (no truncation)


# Printing options using Pandas
def printDataFrame(data_frame, option=1, count=5):
    if option == 1:
        print(data_frame.head(count))  # Prints the dataframe's first n rows
    if option == 2:
        print(data_frame.tail(count))  # Prints the dataframe's last n rows
    if option == 3:
        print(data_frame.shape)  # Prints the dataframe's row & column count
    if option == 4:
        print(data_frame.isnull().sum())  # Prints the dataframe's null value count for each column


# Request a specific attribute for the customer's single
def requestAttribute(*args):
    question_label = None  # Initialize column name
    question_clarity = None  # Provides example for needed user input

    # Assign the correct column name for the output question
    for option in args:
        if option == 'track_artist_name':
            question_label = "artist's name"
            question_clarity = "The Beatles, Michael Jackson, Taylor Swift"
            pass
        elif option == 'track_name':
            question_label = "song title"
            question_clarity = "Hey Jude, Thriller, Shake It Off"
            pass
        elif option == 'artist_genres':
            question_label = "artist genre(s)"
            question_clarity = "Rock, Pop, Country, Rap (Separate by commas if multiple genres are inputted)"
            pass
        else:
            print(f"Invalid option: {option}")
            return

    # Receive the track artist's name as input
    while True:
        user_input = input(f"\nWhat is the {question_label}?\n"
                           f"\nExample Input: {question_clarity}\n"
                           "\nEnter here: ")
        if user_input.strip():
            new_attribute = user_input
            break
        else:
            print("\nNothing was entered! Please enter the requested information.")

    return new_attribute


def testPlaylistGenerator(playlist_generator):
    # Start Statement
    print("Testing Initiated.\n")

    # TEST 1: Cleaning Attribute values
    print("\nTEST 1: Cleaning Attribute Values")
    print(f"    Before: Customer single -> {playlist_generator.customer_single_DF['artist_genres'].iloc[0]}")
    print(f"    Before: Song dataset    -> {playlist_generator.song_dataset_DF['artist_genres'].iloc[0]}")
    playlist_generator.cleanAttribute(True, True, True)
    print(f"    After:  Customer single -> {playlist_generator.customer_single_DF['artist_genres'].iloc[0]}")
    print(f"    After:  Song dataset    -> {playlist_generator.song_dataset_DF['artist_genres'].iloc[0]}")

    # TEST 2: Tokenize Attribute values
    print("\nTEST 2: Tokenize Attribute Values")
    result = playlist_generator.song_dataset_DF['artist_genres'].iloc[0]
    print(f"    Before: Type:, {type(result)} -> {result}")
    result = playlist_generator.setTokenizedAttribute()
    print(f"    After:  Type:, {type(result)} -> {result[0]}")

    # TEST 3: Vectorize Tokens by Creating a TF-IDF Feature Matrix
    print("\nTEST 3: Vectorize Tokens by Creating a TF-IDF Feature Matrix")
    result = playlist_generator.getTokenizedAttribute()
    print(f"    Before: Type:, {type(result)} -> {result[0]}")
    result = playlist_generator.setAttributeMatrix()
    print(f"    After:  Type:, {type(result)} -> {result[0]}")

    # TEST 4: Calculate Cosine Similarity of All Vectorized Attribute Pairs
    print("\nTEST 4: Calculate Cosine Similarity of All Vectorized Attribute Pairs")
    result = playlist_generator.setAttributeMatrix()
    print(f"    Before: Type:, {type(result)} -> {result[0]}")
    result = playlist_generator.setSimilarityMatrix()
    print(f"    After:  Type:, {type(result)} -> {result[0]}")

    # TEST 5: Generate Dictionary of Songs Similar to the Customer Single
    print(f"\nTEST 5: Generate Dictionary of Songs Similar to the Customer Single")
    track_artist_name = playlist_generator.customer_single_DF['track_artist_name'].iloc[0]
    track_name = playlist_generator.customer_single_DF['track_name'].iloc[0]
    artist_genres = playlist_generator.customer_single_DF['artist_genres'].iloc[0]
    print(f"\n    Customer Single: {track_name} by {track_artist_name}"
          f"\n    Customer Genres: {artist_genres}")
    playlist_generator.setSongDictionary()

    min_count = 0
    curr_count = min_count
    max_count = 2
    for song_id in playlist_generator.similarity_scores:
        if curr_count == min_count:
            curr_count += 1
            continue
        track_artist_name = playlist_generator.combined_data_frame.iloc[song_id[0]]['track_artist_name']
        track_name = playlist_generator.combined_data_frame.iloc[song_id[0]]['track_name']
        artist_genres = playlist_generator.combined_data_frame.iloc[song_id[0]]['artist_genres']
        print(f"\n    Recommended Single #{curr_count}: {track_name} by {track_artist_name}"
              f"\n    Recommended Genres #{curr_count}: {artist_genres}")
        curr_count += 1
        if curr_count == max_count:
            break

    # Complete Statement
    print("\nTesting Complete.")
