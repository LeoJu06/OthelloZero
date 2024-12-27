from fpdf import FPDF
import json
from operator import itemgetter

def load_trello_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def parse_cards(cards):
    workflow = []
    total_estimated_time = 0
    total_actual_time = 0

    for card in cards:
        card_info = {
            "name": card.get("name"),
            "description": card.get("desc", "Nicht benötigt"),
            "dateLastActivity": card.get("dateLastActivity"),
            "checklist": []
        }
        print(card_info["name"])
        print(card_info["description"])
        checklists = card.get("checklists", [])
        for checklist in checklists:
            items = []
            for item in checklist.get("checkItems", []):
                item_name = item.get("name", "")
                
                if "Geschätzte Bearbeitungszeit" in item_name:
                    try:
                        total_estimated_time += float(''.join(filter(str.isdigit, item_name)))
                    except ValueError:
                        pass
                elif "Benötigte Bearbeitungszeit" in item_name:
                    try:
                        total_actual_time += float(''.join(filter(str.isdigit, item_name)))
                    except ValueError:
                        pass
                items.append({
                    "name": item_name,
                    "state": item.get("state")
                })
            card_info["checklist"].append({
                "name": checklist.get("name"),
                "items": items
            })
        workflow.append(card_info)

    workflow.sort(key=itemgetter("dateLastActivity"), reverse=True)
    return workflow, total_estimated_time, total_actual_time

def create_pdf_workflow(file_path, output_file):
    trello_data = load_trello_data(file_path)
    cards = trello_data.get("cards", [])
    workflow, total_estimated_time, total_actual_time = parse_cards(cards)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Fonts
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    pdf.add_font('DejaVu', '', font_path, uni=True)
    pdf.add_font('DejaVu', 'B', font_path, uni=True)
    pdf.add_font('DejaVu', 'I', font_path, uni=True)
    pdf.add_font('DejaVu', 'BI', font_path, uni=True)

    # Title Page
    pdf.set_font('DejaVu', 'B', size=16)
    pdf.set_text_color(0, 102, 204)  # Blue
    pdf.cell(200, 10, txt="Workflow Report", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font('DejaVu', '', size=12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(200, 10, txt=f"Total Estimated Time: {total_estimated_time} minutes", ln=True)
    pdf.cell(200, 10, txt=f"Total Actual Time: {total_actual_time} minutes", ln=True)
    pdf.ln(10)

    for step in workflow:
        # Task Title
        pdf.set_font('DejaVu', 'B', size=12)
        pdf.set_fill_color(230, 230, 230)  # Light Gray
        pdf.cell(0, 10, txt=f"Task: {step['name']}", ln=True, fill=True)

        # Task Description
        pdf.set_font('DejaVu', '', size=12)
        pdf.multi_cell(0, 10, txt=f"Description: {step['description']}")

        # Checklist Items
        for checklist in step["checklist"]:
            pdf.set_font('DejaVu', 'I', size=12)
            pdf.set_text_color(100, 100, 100)  # Gray
            pdf.multi_cell(0, 10, txt=f"  Checklist: {checklist['name']}")
            for item in checklist["items"]:
                pdf.set_font('DejaVu', '', size=12)
                pdf.set_text_color(0, 0, 0)
                pdf.multi_cell(0, 10, txt=f"    - {item['name']} (Status: {item['state']})")
        pdf.ln(5)
        pdf.set_draw_color(200, 200, 200)  # Light Gray
        pdf.cell(0, 0, txt="", ln=True, border='T')  # Separator line
        pdf.ln(5)

    pdf.output(output_file)

if __name__ == "__main__":
    create_pdf_workflow("data/trello_board.json", "workflow_report_enhanced.pdf")
