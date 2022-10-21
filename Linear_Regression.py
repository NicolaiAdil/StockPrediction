import pandas as pd

def getData(csvFile : str):
    df = pd.read_csv(csvFile)
    
    date = df['Date'].tolist()
    close = df['Close'].tolist()

    return (date, close)

def main():
    csvFile = "GjF_OneYear.csv"
    date, close = getData(csvFile)
    return



if __name__ == "__main__":
    main()