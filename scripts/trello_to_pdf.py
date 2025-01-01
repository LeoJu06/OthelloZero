import json
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime


def extract_trello_data(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    tasks = []

    for card in data["cards"]:
        task = {
            "Name": card.get("name", ""),
            "Description": card.get("desc", ""),
            "Comments": [],
            "Checklists": {},
            "LastActivity": card.get("dateLastActivity", "1970-01-01T00:00:00.000Z"),
        }

        # Extract actions (comments)
        for action in card.get("actions", []):
            if action.get("type") == "commentCard":
                task["Comments"].append(action["data"].get("text", ""))

        # Extract checklists
        for checklist in card.get("checklists", []):
            for item in checklist.get("checkItems", []):
                if "geschätzte" in item.get("name", "").lower():
                    task["Checklists"]["Geschätzte Bearbeitungszeit"] = item.get(
                        "name", ""
                    )
                elif "benötigte" in item.get("name", "").lower():
                    task["Checklists"]["Benötigte Bearbeitungszeit"] = item.get(
                        "name", ""
                    )

        tasks.append(task)

    # Sort tasks by LastActivity in descending order
    tasks.sort(
        key=lambda x: datetime.fromisoformat(x["LastActivity"].replace("Z", "+00:00")),
        reverse=True,
    )

    return tasks


def wrap_text(text, font, font_size, max_width, canvas):
    """Wrap text to fit within a specified width."""
    words = text.split(" ")
    lines = []
    current_line = []
    for word in words:
        test_line = " ".join(current_line + [word])
        text_width = canvas.stringWidth(test_line, font, font_size)
        if text_width <= max_width:
            current_line.append(word)
        else:
            lines.append(" ".join(current_line))
            current_line = [word]
    if current_line:
        lines.append(" ".join(current_line))
    return lines


def create_pdf(tasks, output_file):
    c = canvas.Canvas(output_file, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.setFillColorRGB(0.2, 0.4, 0.8)  # Blue color for title
    c.drawCentredString(width / 2, height - 40, "Trello Workflow Tasks")
    c.setFillColorRGB(0, 0, 0)  # Reset to black for body text

    y_position = height - 80
    max_width = width - 60  # Allow margins on the sides

    for task in tasks:
        if y_position < 100:  # Page break if near the bottom of the page
            c.showPage()
            y_position = height - 40

        # Task Name
        c.setFont("Helvetica-Bold", 12)
        task_name = f"Task Name: {task['Name']}"
        lines = wrap_text(task_name, "Helvetica-Bold", 12, max_width, c)
        for line in lines:
            c.drawString(30, y_position, line)
            y_position -= 15
        y_position -= 5  # Extra spacing

        # Description
        c.setFont("Helvetica", 10)
        description = f"Description: {task['Description']}"
        lines = wrap_text(description, "Helvetica", 10, max_width, c)
        for line in lines:
            c.drawString(30, y_position, line)
            y_position -= 15
        y_position -= 5  # Extra spacing

        # Comments
        c.setFont("Helvetica-Bold", 10)
        c.drawString(30, y_position, "Comments:")
        y_position -= 15

        c.setFont("Helvetica", 10)
        if task["Comments"]:
            for comment in task["Comments"]:
                lines = wrap_text(f"- {comment}", "Helvetica", 10, max_width, c)
                for line in lines:
                    c.drawString(30, y_position, line)
                    y_position -= 15
        else:
            c.drawString(30, y_position, "No comments.")
            y_position -= 15
        y_position -= 5  # Extra spacing

        # Checklists
        c.setFont("Helvetica-Bold", 10)
        c.drawString(30, y_position, "Checklists:")
        y_position -= 15

        c.setFont("Helvetica", 10)
        if task["Checklists"]:
            for key, value in task["Checklists"].items():
                checklist_line = f"- {key}: {value}"
                lines = wrap_text(checklist_line, "Helvetica", 10, max_width, c)
                for line in lines:
                    c.drawString(30, y_position, line)
                    y_position -= 15
        else:
            c.drawString(30, y_position, "No checklists.")
            y_position -= 15

        # Add spacing between tasks
        y_position -= 20

    c.save()


# Example usage
file_path = "data/trello_board.json"
tasks = extract_trello_data(file_path)

# Output PDF file
output_file = "Trello_Worflow.pdf"
create_pdf(tasks, output_file)

print(f"PDF has been generated and saved as {output_file}")
