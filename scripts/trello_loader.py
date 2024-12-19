import requests
import json
from dotenv import load_dotenv
import os

# Load environment variables from the .env file to securely access sensitive information
load_dotenv()

# Retrieve the Trello API key, token, and board ID from environment variables
# These values should be stored securely and not hardcoded in the script
api_key = os.getenv("API_KEY")
token = os.getenv("TOKEN")
board_id = os.getenv("BOARD_ID")


def fetch_trello_board_json(api_key, token, board_id):
    """
    Fetch detailed information about a Trello board using the Trello API.

    This function makes a GET request to the Trello API to retrieve various details of a board,
    such as lists, cards, labels, members, checklists, custom fields, and actions. The data is
    returned as a dictionary if the request is successful, or None if an error occurs.

    Args:
        api_key (str): The Trello API key used for authentication.
        token (str): The Trello API token used for authentication.
        board_id (str): The ID of the Trello board from which data will be fetched.

    Returns:
        dict or None: A dictionary containing the board data if the request is successful,
                      or None if there was an error with the request.
    """
    # Define the URL for the Trello API endpoint to fetch board details
    url = f"https://api.trello.com/1/boards/{board_id}"

    # Set up the parameters to specify which data to fetch from the board
    params = {
        'key': api_key,          # API key for authentication
        'token': token,          # API token for authentication
        'fields': 'all',         # Request all fields related to the board (lists, cards, etc.)
        'lists': 'all',          # Include all lists in the board
        'cards': 'all',          # Include all cards within the board
        'labels': 'all',         # Include all labels assigned to cards
        'member': 'true',        # Include member information (who is assigned to cards)
        'checklists': 'all',     # Include all checklists within cards
        'customFields': 'true',  # Include all custom fields defined for the board
        'actions': 'all',        # Include all actions (history of changes) related to the board
    }

    try:
        # Make a GET request to the Trello API to fetch the board's data
        response = requests.get(url, params=params)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            return response.json()  # Return the board data as a JSON object
        else:
            # Print an error message if the request was not successful
            print(f"Error fetching the board: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        # Catch any network-related or request errors and print the error message
        print(f"An error occurred: {e}")
        return None


def save_json_to_file(data, filename):
    """
    Save the provided data to a JSON file with proper formatting.

    This function creates any necessary directories if they don't already exist
    and writes the provided data to the specified file in a human-readable format.

    Args:
        data (dict): The data to be saved to the JSON file.
        filename (str): The path of the file where the data will be saved.

    Returns:
        None
    """
    # Ensure the directory for the file exists, and create it if necessary
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Open the file in write mode and save the data as a pretty-printed JSON
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

    # Print a message confirming that the data has been saved successfully
    print(f"JSON data has been saved to {filename}")


def main():
    """
    Main function to load a Trello board's data and save it to a JSON file.

    This function calls the `fetch_trello_board_json` function to retrieve the board's data
    and then saves it to a file using the `save_json_to_file` function. The file is saved in the
    'data' directory with the filename 'trello_board.json'.

    Returns:
        None
    """
    # Fetch the Trello board's data using the provided API key, token, and board ID
    board_json = fetch_trello_board_json(api_key, token, board_id)

    # If the board data was successfully fetched, save it to a JSON file
    if board_json:
        save_json_to_file(board_json, "data/trello_board.json")


# This ensures the main function is executed only when the script is run directly
if __name__ == "__main__":
    main()
