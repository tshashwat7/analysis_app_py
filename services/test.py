from services.corporate_actions import get_corporate_actions

# Simulate the Name Map you have in main.py
mock_map = {
    "INFY.NS": "Infosys",
    "TCS.NS": "Tata Consultancy Services",
    "RELIANCE.NS": "Reliance Industries"
}

print("Fetching...")
# This will scrape MoneyControl and cache it
res = get_corporate_actions(["INFY.NS", "TCS.NS"], mode="upcoming", name_map=mock_map)

import json
print(json.dumps(res, indent=2))