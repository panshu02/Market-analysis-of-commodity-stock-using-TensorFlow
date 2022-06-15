import pandas as pd
import requests
from bs4 import BeautifulSoup


def give_df(url):
    response = requests.get(url)
    if response.status_code == 200:
        print("**\tData imported successfully\t**\n\n")
    else:
        print("**\tUnable to import data\t**\n\n")

    response = BeautifulSoup(response.text, 'html.parser')

    table = response.find('div', {'id' : 'results_box'}).find_all('tr')

    cols = table[0].find_all('th')
    n_cols = len(cols)

    columns = [cols[i].text for i in range(n_cols)]

    data = pd.DataFrame(columns=columns)

    for i in range(1, len(table)-1):
        row = table[i].find_all('td')
        tuple = []
        for j in range(n_cols):
            tuple.append(row[j].text)
        data.loc[i-1] = tuple
    
    return data
