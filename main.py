import pandas as pd
import requests
from io import BytesIO

api_url = 'https://api.dane.gov.pl/1.4/resources/276637,baza-ruchu-granicznego-styczen-wrzesien-2025?lang=pl'
response = requests.get(api_url)
response.raise_for_status()
data = response.json()
download_url = data['data']['attributes']['download_url']

print('Download URL :', download_url)
file_response = requests.get(download_url)

xls = pd.ExcelFile(BytesIO(file_response.content))
print('Sheet : ', xls.sheet_names)

df = pd.read_excel(xls, sheet_name= xls.sheet_names[0])
df.to_csv("data/raw/frame_1.csv", index = False)
print(df)

