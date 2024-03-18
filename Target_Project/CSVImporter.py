import pandas as pd

class CSVDataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.summary = None

    def import_data(self):
        try:
            self.data = pd.read_csv(self.file_path)
            self.summary = "Data imported successfully.\n"
        except FileNotFoundError:
            self.summary = "File not found. Please provide a valid file path.\n"

    def handle_missing_data(self):
        if self.data is not None:
            missing_values = self.data.isnull().sum()
            self.summary += "Variable summaries:\n"
            for column in self.data.columns:
                self.summary += f"Variable: {column}\n"
                self.summary += f"Data type: {self.data[column].dtype}\n"
                self.summary += f"Missing entries: {missing_values[column]}\n"

                # Fill missing values with 'N/A' for object columns
                if self.data[column].dtype == 'object':
                    self.data[column].fillna('N/A', inplace=True)

                # Convert time in HH:MM format to total minutes since midnight
                if self.data[column].dtype == 'object' and self.data[column].str.match(r'\d{2}:\d{2}').all():
                    self.data[column] = pd.to_datetime(self.data[column], format='%H:%M').dt.hour * 60 + pd.to_datetime(self.data[column], format='%H:%M').dt.minute

        else:
            self.summary = "No data imported yet. Please import data first.\n"

    def generate_summary(self):
        if self.data is not None:
            return self.summary
        else:
            return "No summary available. Please import data first."