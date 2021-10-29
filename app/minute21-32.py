#####################################################################
#                                                                   #
#  FROM THE 21st MEETING (28011998) TO THE 32nd MEETING (01011999)  #
#                                                                   #
#####################################################################

from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd

# Abrir a URL que contém as atas das reuniões do COPOM e obter o seu código-fonte.

print("Executing Code")
url_publications = 'https://www.bcb.gov.br/publicacoes/atascopom/cronologicos'
option = Options()
option.headless = True
print("Opening Driver")
driver = webdriver.Firefox(options=option)
#driver = webdriver.Firefox(executable_path = r'C:\(...)\geckodriver.exe') # se o executável (.exe) não estiver no $PATH do Windows.
print("Getting", url_publications)
driver.get(url_publications)
print("Waiting Page Loading")
WebDriverWait(driver, 20).until(EC.presence_of_element_located(
    (By.XPATH, "//div[@class='resultados-relacionados row']//a[@class='btn btn-xs btn-vermais btn-color-1']")))
html = driver.page_source
print("Page Source Ok")
driver.quit()

# Extrair os dados (parsear) do arquivo HTML da URL que contém os caminhos das atas e transformá-los em um dado estruturado.

print("Loading Parser")
soup_publications = BeautifulSoup(html, "html.parser")
table = soup_publications.find("div", "resultados-relacionados row")
a_tag = table.find_all("a", "btn btn-xs btn-vermais btn-color-1")
print("Parsing Completed")

# Criar Lista dos Caminhos das atas.

path_minutes = []
print("Creating List of Path Minutes")
for hyperlink in a_tag:
    link_path = hyperlink.get('href')
    path_minutes.append(link_path)
print("List Created")
#print(path_minutes)

# Criar Lista das URL's.

url_base = 'https://www.bcb.gov.br'
url_minutes = []
print("Creating List of URL's")
for each_path in path_minutes:
    url_minutes.append(f"{url_base}{each_path}")
print("List Created")
#print(url_minutes) 

# Obter o HTML de cada página com o conteúdo das atas e Criar uma Lista com os respectivos Códigos Fontes.

driver_option = Options()
driver_option.headless = True
sources = []
print("Creating List of Sources")
print("Opening Driver")
with webdriver.Firefox(options=driver_option) as driver:
    for each_url in url_minutes[-12:]:
        print("Getting", each_url)
        driver.get(each_url)
        print("Waiting Page Loading")
        WebDriverWait(driver, 20).until(EC.presence_of_element_located(
            (By.XPATH, "//div[@class='container main']//div[@class='col-md-12 col-xl-12']//div[@class='row']")))
        sources.append(driver.page_source)
        print("Page Source Ok")
driver.quit()
print("List Created")
#print(sources)

# Fazer o Parse de cada página HTML.                        

pages_text = []
print("Loading Parser for Each HTML Page")
for each_page in sources:
    soup_text = BeautifulSoup(each_page, "html.parser")
    text_body = soup_text.find("div", {"class": "container main"})
    paragraphs = text_body.find_all("p")
    texts = []
    for p in paragraphs:
        texts.append(p.text)
    text_together = " ".join(texts)
    text_together = text_together.strip()
    for ch in ['\n', '\xa0', '\xad ', '  ', '   ', '    ',
               '     ' , '      ', '       ']:
        if ch in text_together:
            text_together = text_together.replace(ch, " ")
    pages_text.append(text_together)
print("Parsing Completed")
#print(pages_text)

# Salvar em CSV.

print("Saving CSV")
df = pd.DataFrame(pages_text)
df.columns = ['Minutes']
df.to_csv('minute21-32.csv', index=False)
print("CSV Saved")
print("Code Executed")
