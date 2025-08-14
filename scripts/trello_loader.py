import requests
import json
from dotenv import load_dotenv
import os

path_to_json_file = "data/trello_board.json"

# Load environment variables from the .env file to securely access sensitive information
load_dotenv()

# Retrieve the Trello API key, token, and board ID from environment variables
api_key = os.getenv("API_KEY")
token = os.getenv("TOKEN")
board_id = os.getenv("BOARD_ID")


def fetch_trello_board_json(api_key, token, board_id):
    """
    Fetch detailed information about a Trello board using the Trello API, with pagination.
    """
    # Define the URL for the Trello API endpoint to fetch board details
    url = f"https://api.trello.com/1/boards/{board_id}"

    # Set up the parameters to specify which data to fetch from the board
    params = {
        "key": api_key,  # API key for authentication
        "token": token,  # API token for authentication
        "fields": "all",  # Request all fields related to the board (lists, cards, etc.)
        "lists": "all",  # Include all lists in the board
        "labels": "all",  # Include all labels assigned to cards
        "members": "true",  # Include member information (who is assigned to cards)
        "checklists": "all",  # Include all checklists within cards
        "customFields": "true",  # Include all custom fields defined for the board
        "actions": "all",  # Include all actions (history of changes) related to the board
    }

    all_cards = []
    page = 1
    limit = 100  # Set the limit for the number of cards per page
    i = 0

    while True:
        # Set the pagination parameters for cards
        params["limit"] = limit
        params["page"] = page

        # Make a GET request to the Trello API to fetch the board's data
        response = requests.get(f"{url}/cards", params=params)

        if response.status_code == 200:
            print(f"Received cards successfully {page}.")
            cards = response.json()
            all_cards.extend(cards)

            # If fewer than 'limit' cards are returned, we have reached the end
            if len(cards) < limit:
                break

            if page >= 1:
                print("Reached the maximum number of pages (5). Stopping further requests.")
                break

            # Otherwise, move to the next page
            page += 1


        else:
            print(f"Error fetching cards: {response.status_code} - {response.text}")
            break

    # Fetch lists, labels, members, etc.
    lists_response = requests.get(f"{url}/lists", params=params)
    labels_response = requests.get(f"{url}/labels", params=params)
    members_response = requests.get(f"{url}/members", params=params)

    # Check if these requests were successful and retrieve the data
    lists = lists_response.json() if lists_response.status_code == 200 else []
    labels = labels_response.json() if labels_response.status_code == 200 else []
    members = members_response.json() if members_response.status_code == 200 else []

    # Combine all data in a single dictionary
    board_data = {
        "cards": all_cards,
        "lists": lists,
        "labels": labels,
        "members": members,
    }

    return board_data


def save_json_to_file(data, filename):
    """
    Save the provided data to a JSON file with proper formatting.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Save data to JSON file
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

    print(f"JSON data has been saved to {filename}")


def main():
    """
    Main function to load a Trello board's data and save it to a JSON file.
    """
    # Fetch the Trello board's data using the provided API key, token, and board ID
    board_json = fetch_trello_board_json(api_key, token, board_id)

    # If the board data was successfully fetched, save it to a JSON file
    if board_json:
        save_json_to_file(board_json, path_to_json_file)


def load_trello_json():
    with open(path_to_json_file, "r") as file:
        data = json.load(file)

    return data


if __name__ == "__main__":
    main()
