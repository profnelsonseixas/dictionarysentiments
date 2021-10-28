from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

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
    for each_url in url_minutes[-12:]:  ## iteração pelas primeiras três páginas.
        print("getting", each_url)
#        url_base_source = each_url
#        driver.get(url_base_source)
        driver.get(each_url)
        print("waiting for text")
        WebDriverWait(driver, 20).until(EC.presence_of_element_located(
            (By.XPATH, "//div[@class='container main']//div[@class='col-md-12 col-xl-12']//div[@class='row']")))  # espera até que o texto seja carregado na página.
        sources.append(driver.page_source)
        print("ok")
driver.quit()
#print(sources)

# Fazer o Parse de cada página HTML.                        

pages_text = []
for each_page in sources:
    soup_text = BeautifulSoup(each_page, "html.parser")
    text_body = soup_text.find("div", {"class": "container main"})  # Encontra o corpo do texto
    paragraphs = text_body.find_all("p")  # Encontra cada paragrafo
    texts = []
    for p in paragraphs:  # Itera sobre os paragrafos, pegando o texto de cada um
        texts.append(p.text)
    text_together = " ".join(texts) # o text_together é da classe string (str).
    text_together = text_together.strip()
    for ch in ['\n', '\xa0', '\xad ', '  ', '   ', '    ',
               '     ' , '      ', '       ']:
        if ch in text_together:
            text_together = text_together.replace(ch, " ")
    pages_text.append(text_together) # "torna" o text_together novamente uma lista - cada ata vira um elemento da lista pages_text.

print(pages_text)

# Limpeza restante: retirar a string '\', pois tem que tomar cuidado para não excluir a string 'Moody\'s'
# Também verificar algumas palavras que ficaram unidas na Ata 21
