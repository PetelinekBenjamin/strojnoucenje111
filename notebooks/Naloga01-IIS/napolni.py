import shutil

source_file = "data/processed/GOSPOSVETSKA C - TURNERJEVA UL.csv"
destination_file = "data/processed/reference_data.csv"

# Kopiraj in prepisi vsebino iz vira v cilj
shutil.copyfile(source_file, destination_file)

print("Datoteka uspešno prepisana.")
