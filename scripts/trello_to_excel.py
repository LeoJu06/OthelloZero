import json
import pandas as pd
from openpyxl import Workbook
from trello_loader import path_to_json_file, load_trello_as_json  # Hier verwenden wir die Pfadvariable aus trello_loader
import trello_loader  # Import der trello_loader-Bibliothek, die das JSON lädt


def create_excel_from_trello_json(trello_json, output_file="trello_workflow.xlsx"):
    """
    Creates an Excel sheet from a Trello JSON containing information about cards, their movements,
    comments, estimated and actual time, and dates.

    Args:
        trello_json (dict): The Trello board data in JSON format.
        output_file (str): The file path where the Excel sheet will be saved. Default is "trello_data.xlsx".

    Returns:
        None: This function saves the Excel file to the specified path.
    """

    # Initialize an empty list to store card data for each row
    data = []

    # Extract relevant data for each card
    for card in trello_json.get("cards", []):
        # Extract card name
        card_name = card.get("name", "No Name")
        
        # Get the list the card is currently in (if available)
        current_list = card.get("list", {}).get("name", "No List")
        
        # Get comments (activity) associated with the card
        comments = []
        for action in trello_json.get("actions", []):
            if action.get("data", {}).get("card", {}).get("id") == card.get("id"):
                if action.get("type") == "commentCard":
                    comments.append(action.get("data", {}).get("text", "No comment"))
        
        # Extract estimated time and actual time (custom field values, if present)
        estimated_time = None
        actual_time = None
        for field in card.get("customFields", []):
            if field.get("name") == "Geschätzte Zeit":
                estimated_time = field.get("value", {}).get("text", "Not Set")
            if field.get("name") == "Benötigte Zeit":
                actual_time = field.get("value", {}).get("text", "Not Set")
        
        # Get the date from actions (if available)
        action_dates = []
        for action in trello_json.get("actions", []):
            if action.get("data", {}).get("card", {}).get("id") == card.get("id"):
                action_dates.append(action.get("date"))

        # Collect all information into a row
        data.append({
            "Card Name": card_name,
            "Current List": current_list,
            "Comments": "\n".join(comments) if comments else "No Comments",
            "Estimated Time": estimated_time,
            "Actual Time": actual_time,
            "Date": ", ".join(action_dates) if action_dates else "No Date"
        })

    # Create a DataFrame from the collected data
    df = pd.DataFrame(data)

    # Write the data to an Excel file
    df.to_excel(output_file, index=False, engine='openpyxl')

    print(f"Excel file has been created and saved as '{output_file}'")


if __name__ == "__main__":
    # Läd das JSON vom Trello Board durch die trello_loader main Funktion
    trello_loader.main()

    # Öffne das JSON, das durch trello_loader heruntergeladen wurde
    trello_json = load_trello_as_json()
    
    # Rufe die Funktion auf, um die Excel-Datei zu erstellen
    create_excel_from_trello_json(trello_json)

