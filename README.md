# AI Music Playlist Generator
WGU C964 Computer Science Capstone Project

## SCENARIO
A music marketing company has experienced an increase in its customer base and is having issues providing high quality, custom-curated playlists for its customers’ new singles in a timely manner because playlists are curated manually and employees are unable to keep up with demand. As a software engineer, I propose to design a software solution that utilizes machine learning to automatically generate a playlist of songs that best match a customer’s given single through the use of the company’s current dataset of popular songs that employees use to manually construct playlists.

This data product I am proposing will benefit the customer and support the decision-making process for playlist generation. The customer will experience a virtually instantaneous response time between single submission and playlist curation given the playlist curation’s automatic processes. The customer only needs to provide song attributes that the software will analyze, the employee will enter such attributes, and the product will return a playlist. This is better than the customer having to consult with an employee, discuss certain qualities that make up their song, or wait for an employee to manually match the customer’s song with songs in our dataset. Also, such a data product will allow standardization in what can be considered a well-constructed playlist for a given customer’s single. For example, the data product can prioritize artist genre. Such standards can remove bias that customers occasionally experience from the current manual curation process.

## COMPETENCIES
1. Write Python Code
2. Implement Machine Learning
3. Solve Problems for an Organization

## SKILLS
1. Software Engineering
2. Software Development Life Cycle
3. Python
4. Machine Learning
5. Recommendation Systems
6. User Interface Design
7. Dataframe ETL & Analysis

# How to Setup the Project
Install the latest versions of Python & PyCharm and then follow these steps:
1. MacOS: Open terminal and enter 'git clone <SSH_Link>' to clone the project (Note: Replace the link with the repository's SSH link).
2. Create a new project folder to store the cloned application folder in it.
3. In the application folder, locate & move the zipped file "documentation" outside the application folder, store it in the project folder, & unzip it.
4. PyCharm: open the cloned application folder.
5. Run Application: Open main.py and Run the application with main.py as the current file. The UI will display in the command line (i.e.: terminal).

# How to Use the Project
At runtime, the UI will display in the command line, giving the user (i.e: employee) a numbered list of options. The user will enter the number of the option they wish to select and follow any prompts upon request by the interface. The UI validates all inputs by the user to ensure proper execution of the program. The options include: "Generate Playlist", "Data Visualization - Number of Songs per Decade", "Data Visualization - Top 5 Most Frequent Artist Sub-Genres", "Data Visualization - Most Common Keywords in Artist Genres", "Update Customer Single", and "Quit".

# Troubleshooting
1. Application fails to run: This most likely will occur if main.py is not the default configuration for runtime. Update your runtime configuration settings or open main.py and run main.py as the current file.
2. Python libraries are not being read at runtime: This will most likely show an error messaage indicating which library can't be read. If this occurs, open Options.py and observe all imports (ex: lines 1-5). If using PyCharm, hover over any import that has a red underline and import each of them. If you are not using PyCharm, then you would have to read the User Guide's "Application Setup Step 4" at Page 20 of the "task_2_report" located in the zipped documentations directory for further assistance.
3. Punkt Tokenizer fails to be read: Read the error message, which tells you where nltk is attempting to read the Punkt Tokenizer. You should place the nltk_data folder there. This folder already exists in the program and can be copied to any location as needed. Refer to online tutorial if the issue persists.
