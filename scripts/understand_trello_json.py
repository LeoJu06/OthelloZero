
import trello_loader

trello_loader.main()

trello_json = trello_loader.load_trello_as_json()


# Beispiel: Überprüfe die Struktur des Trello-Boards
if isinstance(trello_json, dict):
    print("Dictionnary keys:")
    print(trello_json.keys())
    print()  # Gibt alle Hauptschlüssel des JSON zurück

print(trello_json["cards"])
#print(trello_json["checklists"])
for i in trello_json["checklists"]:
    pass
    #print(i)



"""
{'id': '67647c7cd172296b2d05a2d8', 'name': 'Timetracker', 'idBoard': '672b9bbc97c86d6599920565', 'idCard': '672ba4999c80f9571589384d', 'pos': 16384, 'checkItems': [{'id': '67647c7cd172296b2d05a2d9', 'name': 'Geschätzte Bearbeitungszeit:', 'nameData': {'emoji': {}}, 'pos': 16384, 'state': 'incomplete', 'due': None, 'dueReminder': None, 'idMember': None, 'idChecklist': '67647c7cd172296b2d05a2d8'}, {'id': '67647c7cd172296b2d05a2da', 'name': 'Benötigte Bearbeitungszeit:', 'nameData': {'emoji': {}}, 'pos': 32768, 'state': 'incomplete', 'due': None, 'dueReminder': None, 'idMember': None, 'idChecklist': '67647c7cd172296b2d05a2d8'}]}
"""