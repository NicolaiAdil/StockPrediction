def getData(csvFile : str) -> tuple(list, list):
    time = []
    close = []
    with open(csvFile, 'r') as f:
        for line in f:
            data.append(line.strip().split(','))
    return data