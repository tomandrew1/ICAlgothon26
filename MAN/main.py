import csv

team_name = "RATT"
N = 1

filename = f"{team_name}_round_{N}.csv"

data = [
    ["INSTRUMENT_1", 0.15],
    ["INSTRUMENT_2", 0.10],
    ["INSTRUMENT_3", 0.05],
    ["INSTRUMENT_4", 0.05],
    ["INSTRUMENT_5", 0.20],
    ["INSTRUMENT_6", 0.10],
    ["INSTRUMENT_7", 0.15],
    ["INSTRUMENT_8", 0.05],
    ["INSTRUMENT_9", 0.10],
    ["INSTRUMENT_10", 0.05]
]

with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)

    writer.writerow(["asset", "weight"])

    writer.writerows(data)

print(f"Success! Created file: {filename}")
