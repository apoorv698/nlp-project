from bs4 import BeautifulSoup
import urllib.request
import nltk
content = urllib.request.urlopen('http://newsone-newsreading.appspot.com/sport.html')
content=content.read()
# print(content)
soup = BeautifulSoup(content, 'html.parser')
# print(soup)
import re
# print(soup.get_text(strip=True))
# print(len(soup.get_text(strip=True)))
'''tag = re.finditer('Dhoni', soup)
#print(len(tag))
for f in tag:
    print(f)
    print(f.regs)'''
htm = str(soup)
tag = re.search('Dhoni', htm)
txt = htm[tag.regs[0][0]:]
soup = BeautifulSoup(txt, 'html.parser')
# print(soup.get_text(strip=True))
print(soup.find('a')['href'])
content = urllib.request.Request(soup.find('a')['href'], headers={'User-Agent':'Magic Browser'})
opener = urllib.request.build_opener()
print(BeautifulSoup(opener.open(content).read(),'html.parser').find('div', {'class':'story-details'}).get_text(strip=True))
