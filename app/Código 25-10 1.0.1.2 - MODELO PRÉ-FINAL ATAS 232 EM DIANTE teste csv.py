from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd

# Abrir a URL que contém as atas das reuniões do COPOM e obter o seu código-fonte.

url_publications = 'https://www.bcb.gov.br/publicacoes/atascopom/cronologicos'

option = Options()
option.headless = True
driver = webdriver.Firefox(options=option)
#driver = webdriver.Firefox(executable_path = r'C:\(...)\geckodriver.exe') # se o executável (.exe) não estiver no $PATH do Windows.
driver.get(url_publications)
element = WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.XPATH, "//div[@class='resultados-relacionados row']//a[@class='btn btn-xs btn-vermais btn-color-1']")))
html = driver.page_source
driver.quit()

# Extrair os dados (parsear) do arquivo HTML da URL que contém os caminhos das atas e transformá-los em um dado estruturado.

soup_publications = BeautifulSoup(html, "html.parser")
table = soup_publications.find("div", "resultados-relacionados row")
a_tag = table.find_all("a", "btn btn-xs btn-vermais btn-color-1")

# Criar Lista dos Caminhos das atas.

path_minutes = []
for hyperlink in a_tag:
    link_path = hyperlink.get('href')
    path_minutes.append(link_path)
#print(path_minutes)

# Criar Lista das URL's.

url_base = 'https://www.bcb.gov.br'
url_minutes = []
for each_path in path_minutes:
    url_minutes.append(f"{url_base}{each_path}")
#print(url_minutes) 

# ATÉ AQUI ESTÁ OK!

# Obter o HTML de cada página com o conteúdo das atas e Criar uma Lista com os respectivos Códigos Fontes.

driver_option = Options()
driver_option.headless = True
#driver = webdriver.Firefox(options=driver_option)

sources = []
print("opening driver")
with webdriver.Firefox(options=driver_option) as driver: # carregou as páginas, pra cada data na lista de datas.
    for each_url in url_minutes[:-211]:  ## iteração pelas primeiras três páginas. #[:10]
        print("getting", each_url)
#        url_base_source = each_url
#        driver.get(url_base_source)
        driver.get(each_url)
        print("waiting for text")
        WebDriverWait(driver, 20).until(EC.presence_of_element_located(
            (By.XPATH, "//div[@id='ataconteudo']")))  # espera até que o texto seja carregado na página.
        sources.append(driver.page_source)
        print("ok")
driver.quit()
#print(sources)

# Fazer o Parse de cada página HTML.

pages_text = []
for each_page in sources:
    soup_text = BeautifulSoup(each_page, "html.parser")
    text_body = soup_text.find("div", {"id": "ataconteudo"})  # Encontra o corpo do texto
    paragraphs = text_body.find_all("p")  # Encontra cada paragrafo
    texts = []
    for p in paragraphs:  # Itera sobre os paragrafos, pegando o texto de cada um
        texts.append(p.text)
    text_together = " ".join(texts) # o text_together é da classe string (str).
    text_together = text_together.strip()
    for ch in [' 1.', ' 2.', ' 3.', ' 4.', ' 5.', ' 6.', ' 7.', ' 8.', ' 9.', ' 10.',
           ' 11.', ' 12.', ' 13.', ' 14.', ' 15', ' 16.', ' 17.', ' 18.', ' 19.', ' 20.',
           ' 21.', ' 22.', ' 23.', ' 24.', ' 25', '  ', '   ', '      ']:
        if ch in text_together:
            text_together = text_together.replace(ch, "")
    text_together = text_together.replace('..', '.')
    pages_text.append(text_together) # "torna" o text_together novamente uma lista - cada ata vira um elemento da lista pages_text.

#print(pages_text)

# Salvar em CSV

name = [pages_text] 
dict = {'Ata': name} 
df = pd.DataFrame(dict) 
df.to_csv('teste1.csv')

# Limpeza restante: retirar a string '1.', pois tem que tomar cuidado para não excluir a string '2021.'
