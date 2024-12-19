import requests
import json

from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("API_KEY")
token = os.getenv("TOKEN")
board_id = os.getenv("BOARD_ID")


def fetch_trello_board_json(api_key, token, board_id):
    """
    Lädt das JSON eines Trello-Boards.
    """
    url = f"https://api.trello.com/1/boards/{board_id}"
    params = {
        'key': api_key,
        'token': token,
        'fields': 'name,desc',  # Welche Felder des Boards geladen werden sollen
        'lists': 'all',         # Alle Listen laden
        'cards': 'all'          # Alle Karten laden
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Fehler beim Laden des Boards: {response.status_code} - {response.text}")
        return None

def save_json_to_file(data, filename):
    """
    Speichert JSON-Daten in einer Datei.
    """
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
    print(f"JSON-Daten wurden in {filename} gespeichert.")

def main():
    # Trello Board JSON laden
    board_json = fetch_trello_board_json(api_key, token, board_id)
    
    if board_json:
        # JSON in Datei speichern
        save_json_to_file(board_json, "data/trello_board.json")

if __name__ == "__main__":
    main()
