from time import time
import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV

# Print Top Words' Function

def print_top_words(model, feature_names, n_top_words):
  for topic_idx, topic in enumerate(model.components_):
    print("\n--\nTopic #{}: ".format(topic_idx + 1))
    message = ", ".join([feature_names[i]
                          for i in topic.argsort()[:-n_top_words - 1:-1]])
    print(message)
  print()

# Load the Minutes

print("Loading Dataset")
t0 = time()
df = pd.read_csv('https://raw.githubusercontent.com/profnelsonseixas/dictionarysentiments/initial_commit/app/minutes_concat.csv')
#print(df)
data_samples = df.Minutes.to_list()
#print(data_samples)
print("done in %0.3fs." % (time() - t0))

# Preprocessing I - Removing Numbers, Punctuations and Characters which is Not a Word Character

clean_data_samples = []
for df_text in data_samples:
    df_text  = re.sub("(\\W|\\d|ª)"," ", df_text) 
    df_text = ' '.join(word_minute for word_minute in df_text.split()
                       if len(word_minute) > 1)
    clean_data_samples.append(df_text)
#print(clean_data_samples)

# Preprocessing II - Removing Stop Words

stop_words = set(stopwords.words('portuguese'))
stop_words.update(['a', 'as', 'o', 'os', 'um', 'uma', 'uns',
                   'umas', 'meu', 'teu', 'seu', 'nosso',
                   'vosso', 'seu', 'minha', 'tua', 'sua',
                   'nossa', 'vossa', 'sua', 'meus', 'teus',
                   'seus', 'nossos', 'vossos', 'seus',
                   'minhas', 'tuas', 'suas', 'nossas',
                   'vossas', 'suas', 'a', 'ante', 'após',
                   'até', 'com', 'contra', 'de', 'desde',
                   'em', 'entre', 'para', 'perante', 'por',
                   'per', 'sem', 'sob', 'sobre', 'trás',
                   'ao', 'aos', 'aonde', 'do', 'na', 'pelo',
                   'desse', 'desta', 'naquele', 'como', 'conforme',
                   'consoante', 'segundo', 'mediante', 'durante',
                   'visto', 'e', 'nem', 'mas', 'porém', 'todavia',
                   'contudo', 'e', 'senão', 'entretanto', 'ou', 'ora',
                   'já', 'quer', 'seja', 'logo', 'pois', 'portanto',
                   'assim', 'que', 'pois', 'porque', 'porquanto',
                   'se', 'caso', 'porque', 'como', 'pois', 'que',
                   'porquanto', 'como', 'que', 'qual', 'quanto', 
                   'conforme', 'como', 'consoante', 'segundo',
                   'que', 'embora', 'conquanto', 'porque',
                   'que', 'quando', 'enquanto', 'que', 'se', 'eu',
                   'tu', 'você', 'ele', 'ela', 'nós', 'vós', 'eles',
                   'elas', 'me', 'te', 'o', 'a', 'lhe', 'se',
                   'nos', 'vos', 'os', 'as', 'lhes', 'se',
                   'senhor', 'senhora', 'senhorita', 'você',
                   'vocês', 'Vossa',  'meu', 'minha', 'meus', 'minhas',
                   'teu', 'tua', 'teus', 'tuas', 'seu', 'sua', 'seus',
                   'suas', 'nosso', 'nossa', 'nossos', 'nossas',
                   'vosso', 'vossa', 'vossos', 'vossas', 'seu',
                   'sua', 'seus', 'suas', 'este', 'estes', 'esta',
                   'estas', 'isto', 'esse', 'esses', 'essa',
                   'essas', 'isso', 'aquele', 'aqueles', 'aquela',
                   'aquelas', 'aquilo', 'este', 'estes', 'esta',
                   'estas', 'isto', 'esse', 'esses', 'essa',
                   'essas', 'isso', 'aquele', 'aqueles', 'aquela',
                   'aquelas', 'aquilo', 'algum', 'alguma', 'alguns',
                   'algumas', 'nenhum', 'nenhuma', 'nenhuns',
                   'nenhumas', 'certo', 'certa', 'certos',
                   'certas', 'muito', 'muita', 'muitos', 'muitas',
                   'todo', 'toda', 'todos', 'todas', 'outro',
                   'outra', 'outros', 'outras', 'pouco', 'pouca',
                   'poucos', 'poucas', 'tanto', 'tanta', 'tantos',
                   'tantas', 'vários', 'várias', 'qualquer',
                   'quaisquer', 'bastante', 'bastantes', 'quanto',
                   'quanta', 'quantos', 'quantas', 'tal', 'tais',
                   'qual', 'quais', 'diverso', 'diversa',
                   'diversos', 'diversas', 'algo', 'alguém',
                   'ninguém', 'tudo', 'nada', 'cada', 'outrem',
                   'quem', 'mais', 'menos', 'que', 'quem', 'qual',
                   'quanto', 'cujo', 'cuja', 'cujos', 'cujas',
                   'quanto', 'quanta', 'quantos', 'quantas',
                   'que', 'quem', 'onde', 'como', 'quando',
                   'janeiro', 'fevereiro', 'março', 'abril', 'maio',
                   'junho', 'julho', 'agosto', 'setembro', 'outubro',
                   'novembro', 'dezembro', 'aberto', 'acumulando',
                   'alta', 'apresentaram', 'apresentou', 'aumentou',
                   'base', 'bilhão', 'comparativamente', 'comparação',
                   'comportamento', 'considerando', 'cresceu',
                   'departamento', 'desempenho', 'dessazonalizados',
                   'devido', 'di', 'diretoria', 'diretrizes', 'doméstica',
                   'doze', 'economias', 'elevou', 'estados', 'fundos',
                   'geral', 'ibge', 'igual', 'impacto', 'início', 'livres',
                   'líquida', 'manteve', 'mensal', 'mesma', 'mil', 'milhões',
                   'médias', 'parte', 'partir', 'patamar', 'paulo',
                   'pesquisa', 'pessoas', 'pontos', 'primeiro',
                   'principalmente', 'processo', 'projeção', 'próximos',
                   'recente', 'recuo', 'recuou', 'recursos', 'refletindo',
                   'registraram', 'registrou', 'resultados', 'ritmo',
                   'saldo', 'sazonal', 'segmentos', 'série', 'total',
                   'unidos', 'variações', 'vez', 'viés', 'volume',
                   'índices', 'abaixo', 'alcançou', 'alexandre', 'apesar',
                   'aumentos', 'avaliação', 'bem', 'brasil', 'calculado',
                   'capacidade', 'colegiada', 'comercializáveis', 'conceito',
                   'consultor', 'conta', 'contrato', 'cresceram', 'curto',
                   'data', 'destaque', 'econômico', 'efeito', 'emergentes',
                   'estável', 'favorável', 'fgv', 'global', 'governo',
                   'horário', 'impactos', 'intenções', 'interna', 'itens',
                   'luiz', 'melhora', 'norte', 'níveis', 'número',
                   'observada', 'observado', 'passou', 'percentual',
                   'ponto', 'presentes', 'projeta', 'prospectiva',
                   'refletiu', 'relativamente', 'repasse', 'retração',
                   'reuniões', 'segmento', 'semestre', 'situou', 'sérgio',
                   'tendências', 'terceiro', 'utilização', 'vem', 'ver',
                   'vista', 'última', 'acumulada', 'americano', 'aparadas',
                   'apenas', 'aumentaram', 'avalia', 'br', 'categorias',
                   'considerados', 'contexto', 'continua', 'continuidade',
                   'contratos', 'contribuição', 'dinâmica', 'disso',
                   'elevado', 'equipamentos', 'eua', 'executivo', 'federais',
                   'finanças', 'função', 'físicas', 'hipótese', 'horizontes',
                   'importante', 'importantes', 'indicador', 'líquidos',
                   'margem', 'movimento', 'negativa', 'observados',
                   'observou', 'países', 'pode', 'positiva', 'posição',
                   'recentes', 'secretário', 'seis', 'sido', 'sinais',
                   'sistema', 'suavização', 'uso', 'valores', 'variou',
                   'último', 'anteriores', 'argentina', 'básicos', 'chefes',
                   'cinco', 'conjunto', 'considerado', 'derivados', 'dessa',
                   'dessazonalizada', 'diretos', 'elevações', 'encerrado',
                   'entanto', 'euro', 'exclusão', 'externa', 'faturamento',
                   'fed', 'hipóteses', 'horizonte', 'informações',
                   'jurídicas', 'lado', 'medidas', 'objetivo',
                   'participantes', 'passado', 'primeiros', 'pundek',
                   'relevante', 'rocha', 'superior', 'tende', 'verificou',
                   'cerca', 'civil', 'considera', 'depec', 'despeito',
                   'divulgado', 'eduardo', 'elevados', 'fato', 'favoráveis',
                   'método', 'neste', 'notas', 'permanece', 'regime',
                   'setores', 'têm', 'acentuada', 'acréscimos', 'ampliado',
                   'atingir', 'ações', 'cabe', 'carlos', 'cni',
                   'consolidação', 'desses', 'doméstico', 'exigibilidade',
                   'externos', 'forte', 'ie', 'leva', 'material', 'ordem',
                   'percepção', 'recuaram', 'referência', 'situação',
                   'termos', 'úteis', 'antonio', 'atual', 'autoveículos',
                   'cumprimento', 'desenvolvimentos', 'deste', 'diário',
                   'esperado', 'longos', 'maduras', 'realizou', 'reduziu',
                   'refere', 'regiões', 'altas', 'consecutivo', 'deban',
                   'decisões', 'demab', 'depin', 'japão', 'líquido',
                   'nominal', 'participação', 'país', 'prazos', 'respectivas',
                   'valorização', 'aceleração', 'caiu', 'decisão',
                   'decorreu', 'disponibilidade', 'empresários', 'ficou',
                   'modelo', 'mundial', 'novo', 'particular', 'presidente',
                   'avançou', 'critério', 'deterioração', 'fluxo',
                   'inferior', 'modo', 'móveis', 'novas', 'referentes',
                   'reflete', 'saldos', 'tempo', 'virtude', 'carteira',
                   'convergência', 'deslocou', 'diminuição', 'elevadas',
                   'especialmente', 'moderação', 'períodos', 'reversão',
                   'subiu', 'trajetórias', 'área', 'cenários',
                   'econômicos', 'federal', 'significativa', 'trimestres',
                   'acumulou', 'atingindo', 'domésticos', 'frente',
                   'próximo', 'apresenta', 'continuam', 'correntes',
                   'deverá', 'funds', 'materiais', 'registrada', 'sentido',
                   'andar', 'elevaram', 'futura', 'futuro', 'indica',
                   'matérias', 'modelos', 'mostram', 'possam', 'privado',
                   'quarto', 'semi', 'tendo', 'ter', 'capitais',
                   'corrente', 'excesso', 'mostraram', 'produto',
                   'acréscimo', 'adicionalmente', 'apresentando',
                   'atividades', 'componente', 'dez', 'entende', 'geração',
                   'grupo', 'pessoal', 'quedas', 'recorde', 'registrando',
                   'subjacente', 'variáveis'])

def remove_repeated(stop_words):
    l = []
    for sw in stop_words:
        if sw not in l:
            l.append(sw)
    l.sort()
    return l

stop_words = remove_repeated(stop_words)
#print(stop_words)

# Use tf (raw term count) features for LDA - Bag of Words

print("Extracting tf features for LDA")
tf_vectorizer = CountVectorizer(
    max_df=0.90, min_df=2, stop_words=stop_words
)
t0 = time()

tf = tf_vectorizer.fit_transform(clean_data_samples)
print("done in %0.3fs." % (time() - t0))

# Fit the LDA model

print(
    "\n" * 2,
    "Fitting LDA models with tf features",
)

lda = LatentDirichletAllocation(
    n_components=3,
    learning_method="online",
    random_state=0,
)

t0 = time()
lda.fit(tf)
print("done in %0.3fs." % (time() - t0))

# Document-Topic Matrix and Topic-Word Matrix

doc_topic_matrix = lda.transform(tf)
topic_word_matrix = lda.components_

print(doc_topic_matrix[0:3])
print(topic_word_matrix[0:3])

# Shape of Document-Topic Matrix and Topic-Word Matrix

print('Document-Topic Matrix:' + str(doc_topic_matrix.shape))
print('Topic-Word Matrix:' + str(topic_word_matrix.shape))

# Topics in LDA Model

print("\n Topics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names_out()
print_top_words(lda, tf_feature_names, 50)

# Defining Search Parameters

print("Defining Search Parameters")
t0 = time()
search_params = {'n_components': [1, 2, 3, 4, 5, 10], 'learning_decay': [.7]}
lda = LatentDirichletAllocation()
model = GridSearchCV(lda, param_grid=search_params)
model.fit(tf)
best_lda_model = model.best_estimator_
print("Best Model's Params: ", model.best_params_)
print("Best Log Likelihood Score: ", model.best_score_)
print("Model Perplexity: ", best_lda_model.perplexity(tf))
print("done in %0.3fs." % (time() - t0))
