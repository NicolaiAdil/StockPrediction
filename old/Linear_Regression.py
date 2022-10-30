import pandas as pd

def getData(csvFile : str) -> tuple[list[str], list[str]]:
    close = []
    date = []
    with open(csvFile, 'r') as file:
        content = file.readlines()
        content.pop(0) #Removes the first line
        
        for line in content:
            line = line.split(",")
            date.append(line[0].strip())
            close.append(line[4].strip())

    return date, close
    

def main():
    print(getData("GjF_OneYear.csv"))
    return



if __name__ == "__main__":
    main()