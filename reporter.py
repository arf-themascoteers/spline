class Reporter:
    def __init__(self, file_name):
        self.file_name = file_name

    def write_columns(self, columns):
        with open(self.file_name, 'w') as file:
            file.write(",".join(columns))

    def write_row(self, row):
        with open(self.file_name, 'a') as file:
            file.write(",".join(row))

