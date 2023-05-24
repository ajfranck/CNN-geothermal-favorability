import csv

def convert_values(input_file, output_file):
    conversion_map = {3.0: 4.0, 2.0: 3.0, 1.0: 2.0, 0.0: 1.0}
    
    with open(input_file, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)
        
        for row in rows:
            for i in range(len(row)):
                try:
                    value = float(row[i])
                    if value in conversion_map:
                        row[i] = str(conversion_map[value])
                except ValueError:
                    pass
        
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)

# Example usage:
input_file = 'predictions.csv'
output_file = 'output.csv'
convert_values(input_file, output_file)
