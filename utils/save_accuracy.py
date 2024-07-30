import csv


def append_accuracy(filename, data):
    with open(f'result/{filename}', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(data)
