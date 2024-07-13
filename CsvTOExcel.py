import pandas as pd

# Corrected file path with double backslashes
file_path = 'D:\\Data Analyst\\Prodigy InfoTech\\Task-3\\bank-additional-full.csv'
data = pd.read_csv(file_path, sep=';')

# Save the cleaned data to an Excel file
output_path = 'D:\\Data Analyst\\Prodigy InfoTech\\Task-3\\cleaned_bank_data.xlsx'
data.to_excel(output_path, index=False)
