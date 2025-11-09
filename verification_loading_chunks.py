import pandas as pd

file_path = r"C:\ProjectK\MarketingAdvantage_AI_v1\data\uploads\Businesses_and_catageroies.xlsx"
df = pd.read_excel(file_path, sheet_name=None)  # reads all sheets
total_rows = sum(len(sheet) for sheet in df.values())
print(f"Total rows across sheets: {total_rows}")

for name, sheet in df.items():
    print(f"\nSheet: {name} — {len(sheet)} rows")
    print(sheet.head(5))
