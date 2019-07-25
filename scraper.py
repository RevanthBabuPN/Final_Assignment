import urllib.request
from bs4 import BeautifulSoup
import lxml
import csv

url = 'https://karki23.github.io/Weather-Data/assignment.html'
sauce = urllib.request.urlopen(url)

srccode = BeautifulSoup(sauce,'lxml')
links =[]
for link in srccode.findAll('a'):
    links.append(link.get('href'))

city_names = [link.split('.')[0] for link in links]

links = ['https://karki23.github.io/Weather-Data/'+i for i in links] #To get full path

i = 0
for url in links:
    sauce = urllib.request.urlopen(url)
    soup = BeautifulSoup(sauce,'lxml')
    table = soup.find('table')
    output_rows = []
    for table_row in table.findAll('tr'):
        columns = table_row.findAll(['td','th'])
        output_row =[]
        for column in columns:
            output_row.append(column.text)
        output_rows.append(output_row)
    filename = 'Dataset/'+city_names[i]+'.csv'
    i = i+1
    with open(filename,'wt+',newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(output_rows)
