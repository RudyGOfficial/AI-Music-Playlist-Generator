import subprocess

from Options import customizePrintDisplay, requestAttribute, testPlaylistGenerator
from PlaylistGenerator import PlaylistGenerator

'''-----------------------------------------Hard-Coded Initialized Variables-----------------------------------------'''
song_dataset = "song_dataset.csv"  # Read the dataset CSV file into a pandas DataFrame
demo_customer_single = "demo_customer_single.csv"  # Customer single, represented with a song in the dataset
song_attribute = "artist_genres"  # Song attribute that determines similarity between songs
relevant_attributes = ["track_artist_name", "track_name", "artist_genres"]  # Limits customer single updates to these
playlist_results = "playlist_generator.txt"  # File where playlist results are saved
customizePrintDisplay()  # Allows Pandas to print tables properly in the command-line interface

# Initialize a playlist generator object with the initialized variables
playlist_generator = PlaylistGenerator(song_dataset, demo_customer_single, song_attribute, True)
# testPlaylistGenerator(playlist_generator)  # Tests all methods involved in the recommendation system

'''----------------------------------------Playlist Generator User Interface-----------------------------------------'''
print("\nWelcome to Playlist Generator!\n")

while True:
    print("Please make a selection.\n\n"
          "1: Generate Playlist\n"
          "2: Data Visualization - Number of Songs per Decade\n"
          "3: Data Visualization - Top 5 Most Frequent Artist Sub-Genres\n"
          "4: Data Visualization - Most Common Keywords in Artist Genres\n"
          "5: Update Customer Single\n"
          "0: Quit\n")

    user_input = input("Enter here: ")
    print(f"\nYou selected {user_input}\n")

    if user_input == '1':

        # Load Statement
        print("Building the song list. Please wait...", end='')

        song_info = (f"A recommended song list has been created based on the following customer single:\n\n"
                     f"Artist Name:  {playlist_generator.customer_single_DF['track_artist_name'].iloc[0]}\n"
                     f"Song Name:    {playlist_generator.customer_single_DF['track_name'].iloc[0]}\n"
                     f"Genres:       {playlist_generator.customer_single_DF['artist_genres'].iloc[0]}\n\n")
        # print(song_info)

        # Clean the chosen song attribute (i.e.: column values)
        playlist_generator.cleanAttribute(True, True, True)

        # These playlist generator methods store their results into their respective instance variables in the class
        playlist_generator.setTokenizedAttribute()  # Stores tokenized attribute values
        playlist_generator.setAttributeMatrix()  # Stores an attribute's TF-IDF feature matrix
        playlist_generator.setSimilarityMatrix()  # Stores a similarity matrix of all attribute pairs
        playlist_generator.setSongDictionary()  # Stores songs that are most similar to a chosen song

        # Print the songs recommended to make a playlist with the customer's single
        playlist_info = (f"Build a playlist with these recommended songs for the chosen single:\n\n"
                         f"{playlist_generator.printSongDictionary(20, True)}"
                         f"\nThanks for using Playlist Generator!\n")
        # print(playlist_info)

        # Complete Statement
        print(" Complete.\n")

        # Create/open the playlist generator text file in write mode ('w') & store results
        with (open(playlist_results, 'w') as file):
            file.write(song_info + playlist_info)

        # Open the file with its default application
        try:
            subprocess.Popen(['open', playlist_results])
        except FileNotFoundError:
            print(song_info + playlist_info)
            print(f"Results have been stored in the '{playlist_results}' file.\n")

    elif user_input == '2':
        playlist_generator.dataVisualizations('songsperdecade')
    elif user_input == '3':
        playlist_generator.dataVisualizations('topartistgenres')
    elif user_input == '4':
        playlist_generator.dataVisualizations('artistgenreskeywords')
    elif user_input == '5':
        # Build a blank dictionary based on the customer_single_DF for the new song
        new_row_data = {col: "" for col in playlist_generator.customer_single_DF.columns}

        # Only assign attributes
        for attribute in new_row_data.keys():
            if attribute in relevant_attributes:
                new_row_data[attribute] = requestAttribute(attribute)

        print(f"\nInputs received. The Customer Single will be changed to the following:\n\n"
              f"Artist Name:  {new_row_data['track_artist_name']}\n"
              f"Song Name:    {new_row_data['track_name']}\n"
              f"Genres:       {new_row_data['artist_genres']}\n\n")

        while True:
            user_input = input("Enter '1' to continue or '0' to cancel: ")
            if user_input == '1':
                # Add values to customer single dataframe
                playlist_generator.updateCustomerSingle(new_row_data)

                print("\nUpdate complete! Returning to previous menu.\n")
                break
            elif user_input == '0':
                print("\nUpdate canceled. Returning to previous menu.\n")
                break
            else:
                print("\nInvalid input! Enter the number corresponding to the options below\n")

    elif user_input == '0':
        print("\nClosing program. Goodbye!")
        break
    else:
        print("\nInvalid input! Enter the number corresponding to the options below\n")
