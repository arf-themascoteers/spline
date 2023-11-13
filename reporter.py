class Reporter:
    def __init__(self, file_name):
        self.file_name = file_name

    def write_columns(self, columns):
        with open(self.file_name, 'w') as file:
            file.write(",".join(columns))
            file.write("\n")

    def write_rows(self, rows):
        with open(self.file_name, 'a') as file:
            for row in rows:
                file.write(",".join([f"{x}" for x in row]))
                file.write("\n")

