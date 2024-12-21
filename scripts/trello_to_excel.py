import json
import pandas as pd
import re


def extract_time(time_str):
    """
    Extracts the first decimal number from a given string.

    Args:
        time_str (str): Input string that may contain a time value.

    Returns:
        float: The extracted number as a float, or None if no number is found.
    """
    match = re.search(r"\d+(\.\d+)?", time_str)
    return float(match.group()) if match else None


def parse_trello_actions(file_path, output_file):
    """
    Parses Trello JSON data to extract workflow details and exports them to an Excel file.

    Args:
        file_path (str): Path to the Trello JSON file.
        output_file (str): Path to the output Excel file.

    Returns:
        None
    """
    # Load JSON data from file
    with open(file_path, "r") as file:
        data = json.load(file)

    # Initialize a list to store workflow data
    workflow_data = []

    # Iterate over each action in the JSON
    for action in data.get("actions", []):
        action_type = action.get("type")
        member_creator = action.get("memberCreator", {}).get("fullName", "Unbekannt")
        date = action.get("date")

        card = action.get("data", {}).get("card", {})
        card_name = card.get("name", "Unbekannte Karte")
        card_desc = card.get("desc", "")

        # Handle comments added to cards
        if action_type == "commentCard":
            comment_text = action["data"].get("text", "")
            workflow_data.append(
                {
                    "Aktion": "Kommentar hinzugefügt",
                    "Kartenname": card_name,
                    "Beschreibung": card_desc,
                    "Kommentar": comment_text,
                    "Mitglied": member_creator,
                    "Datum": date,
                    "Geschätzte Zeit": None,
                    "Benötigte Zeit": None,
                }
            )

        # Extract times from checklist items
        checklist_item = action.get("data", {}).get("checkItem", {}).get("name", "")
        estimated_time = (
            extract_time(checklist_item) if "Geschätzte" in checklist_item else None
        )
        required_time = (
            extract_time(checklist_item) if "Benötigte" in checklist_item else None
        )

        # Handle card-related actions
        if action_type in [
            "copyCard",
            "createCard",
            "updateCard",
            "updateCheckItemStateOnCard",
        ]:
            list_name = (
                action.get("data", {}).get("list", {}).get("name", "Keine Liste")
            )
            board_name = (
                action.get("data", {}).get("board", {}).get("name", "Unbekannt")
            )

            # Skip entries with "Keine Liste" and no estimated or required time
            if list_name == "Keine Liste" and not estimated_time and not required_time:
                continue

            workflow_data.append(
                {
                    "Aktion": "Karte erstellt"
                    if action_type == "copyCard"
                    else "Karte verschoben"
                    if action_type == "updateCard"
                    else "Checkliste aktualisiert",
                    "Kartenname": card_name,
                    "Beschreibung": card_desc,
                    "Liste": list_name,
                    "Board": board_name,
                    "Mitglied": member_creator,
                    "Datum": date,
                    "Geschätzte Zeit": estimated_time,
                    "Benötigte Zeit": required_time,
                }
            )

    # Create a DataFrame from the collected data
    df = pd.DataFrame(workflow_data)

    # Convert the 'Datum' column to datetime and remove timezone information
    df["Datum"] = pd.to_datetime(df["Datum"]).dt.tz_localize(None)

    # Sort data by date in descending order
    df = df.sort_values(by="Datum", ascending=False)

    # Export the DataFrame to an Excel file
    df.to_excel(output_file, index=False, sheet_name="Workflow")
    print(f"Workflow successfully exported to {output_file}.")


if __name__ == "__main__":
    # Input and output file paths
    input_file = "data/trello_board_latest.json"
    output_file = "trello_workflow.xlsx"

    # Parse the Trello actions and export workflow data
    parse_trello_actions(input_file, output_file)
