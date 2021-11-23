from time import time
import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV
import numpy as np
from collections import Counter

print("Executing Code", "\n")

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
data_samples = df.Minutes.to_list()
print("Dataset Loaded in %0.3fs." % (time() - t0), "\n")

# Finding the total number of alphanumeric terms

data_samples_div = []
for df_div in data_samples:
    df_div = len(df_div.split())
    data_samples_div.append(df_div)
print(data_samples_div)

print(len(data_samples_div))
sum_words = sum(data_samples_div)
print(sum_words)

# Preprocessing - Removing Numbers, Punctuations, Non-Word Characters and Stop Words

print("Preprocessing Data - Removing Numbers, Non-Word Characters and Stop Words")
t0 = time()
clean_data_samples = []
for df_text in data_samples:
    df_text  = re.sub("(\\W|\\d|ª)"," ", df_text) 
    df_text = ' '.join(word_minute for word_minute in df_text.split()
                       if len(word_minute) > 1)
    clean_data_samples.append(df_text)

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
                   'novembro', 'dezembro', 'acordo', 'adicionalmente',
                   'ainda', 'alcançou', 'alexandre', 'além', 'andar', 'anos',
                   'anteriores', 'antonio', 'apenas', 'apesar',
                   'apontam', 'apresenta', 'apresentando', 'apresentaram',
                   'apresentou', 'atingindo', 'atingir', 'atingiram',
                   'atingiu', 'atual', 'autoveículos', 'avalia', 'avaliação',
                   'avaliou', 'avançou', 'bilhão', 'bilhões', 'br', 'cabe',
                   'calendário', 'carlos', 'cerca', 'chefes', 'cinco',
                   'civil', 'colegiada', 'comparativamente', 'comparação',
                   'comportamento', 'consecutivo', 'considera', 'considerado',
                   'considerados', 'considerando', 'consultor', 'contexto',
                   'continuam', 'continuará', 'data', 'deban', 'decorreu',
                   'demab', 'demais', 'departamento', 'depec', 'depin',
                   'despeito', 'dessa', 'desses', 'destaca', 'destaque',
                   'deste', 'deve', 'deverá', 'devido', 'dez', 'dia', 'dias', 
                   'diretoria', 'diretrizes', 'discussão', 'discutiram',
                   'disso', 'divulgado', 'divulgados', 'dois', 'doze',
                   'eduardo', 'encerrado', 'encontra', 'entanto', 'entende',
                   'escritório', 'especialmente', 'especificamente',
                   'executivo', 'fato', 'ficou', 'finalizado', 'forma',
                   'frente', 'gerin', 'horário', 'igual', 'importante',
                   'importantes', 'indica', 'interanual', 'lado', 'leva',
                   'luiz', 'membros', 'mesma', 'mil', 'milhares', 'milhão',
                   'milhões', 'modo', 'momento', 'mostram', 'mostraram',
                   'movimento', 'mês', 'nesse', 'neste', 'note', 'novas',
                   'novo', 'observada', 'observado', 'observados', 'observou',
                   'parte', 'particular', 'partir', 'passado', 'passou',
                   'paulo', 'percentual', 'pode', 'podem', 'pondera', 'ponto',
                   'pontos', 'possam', 'presidente', 'primeiro', 'primeiros',
                   'principalmente', 'prospectiva', 'prospectivo', 'próximo',
                   'próximos', 'pundek', 'quarto', 'realizou', 'recente',
                   'recentes', 'refere', 'referentes', 'referência',
                   'reflete', 'refletindo', 'refletiu', 'registrada',
                   'registrado', 'registrando', 'registraram', 'registrou',
                   'regiões', 'relativamente', 'respectivamente',
                   'respectivas', 'respeito', 'reunião', 'reuniões', 'rocha',
                   'sazonalmente', 'secretário', 'segue', 'seis', 'semestre',
                   'sendo', 'sentido', 'ser', 'sido', 'significativa',
                   'situou', 'sobretudo', 'sugerem', 'sérgio', 'tendo', 'ter',
                   'terceiro', 'torno', 'totalizaram', 'trilhão', 'trilhões',
                   'trimestre', 'trimestres', 'três', 'têm', 'uso', 'usual',
                   'vem', 'ver', 'verificou', 'vez', 'virtude', 'vista',
                   'visão', 'última', 'último', 'últimos', 'us'])

def remove_repeated(stop_words):
    l = []
    for sw in stop_words:
        if sw not in l:
            l.append(sw)
    l.sort()
    return l

stop_words = remove_repeated(stop_words)
print(stop_words)
print("Preprocessing Completed in %0.3fs." % (time() - t0), "\n")

# Use tf (raw term count) features for LDA - Bag of Words

print("Extracting tf features for LDA")
t0 = time()
tf_vectorizer = CountVectorizer(stop_words=stop_words)
tf = tf_vectorizer.fit_transform(clean_data_samples)
print("Extraction Completed in %0.3fs." % (time() - t0), "\n")

# Print Shape of Matrix Document-Word

print(tf.shape)

# Print Dictionary

print(tf_vectorizer.get_feature_names())

# Count most frequent words

print(tf_vectorizer.vocabulary_)

# Defining Search Parameters

print("Defining Search Parameters")
n_comp = list(range(1, 51))
search_params = {'n_components': n_comp,
                 'learning_decay': [0.55, 0.60, 0.65, 0.70, 0.75,
                                    0.80, 0.85, 0.90, 0.95, 1.00]}
lda_model = LatentDirichletAllocation()
model_selection = GridSearchCV(lda_model, param_grid=search_params)
model_selection.fit(tf)
best_lda_model = model_selection.best_estimator_
print("Best Model's Params: ", model_selection.best_params_)
print("Best Log Likelihood Score: ", model_selection.best_score_)
print("Search Parameters Defined in %0.3fs." % (time() - t0), "\n")

# Creating Results Table

print("Creating Results Table")
model_results = pd.DataFrame(model_selection.cv_results_)
model_results = cv_results.sort_values()

model_results = model_results.rename(columns = {'param_n_components': 'No. Topics',
                                                'param_learning_decay' : 'Learning Decay',
                                                'mean_test_score': 'Mean Test Score',
                                                'rank_test_score': 'Rank'})

model_results = pd.pivot_table(model_results,
                               index = ['No. Topics', 'Learning Decay'],
                               values = ['Mean Test Score', 'Rank'])

print(model_results)
model_results.head()
print("\n", "Results Table Created in %0.3fs." % (time() - t0), "\n")

# Saving Results Table in CSV

print("Saving CSV")
model_results.to_csv('Results_Table.csv')
print("CSV Saved")

# Fit the LDA model

print("Fitting LDA models with tf features")
lda = LatentDirichletAllocation(n_components=1,
                                learning_method="online",
                                learning_decay=0.55,
                                random_state=0)
t0 = time()
lda.fit(tf)
print("Fit Completed in %0.3fs." % (time() - t0), "\n")

# Document-Topic Matrix and Topic-Word Matrix

print("Getting Document-Topic Matrix and Topic-Word Matrix", "\n")
t0 = time()
doc_topic_matrix = lda.transform(tf)
topic_word_matrix = lda.components_

print(doc_topic_matrix[0:3], "\n")
print(topic_word_matrix[0:3], "\n")

# Shape of Document-Topic Matrix and Topic-Word Matrix

print('Document-Topic Matrix:' + str(doc_topic_matrix.shape))
print('Topic-Word Matrix:' + str(topic_word_matrix.shape), "\n")

print("Matrixes Obtained in %0.3fs." % (time() - t0), "\n")

# Topics in LDA Model

print("Topics in LDA Model")
tf_feature_names = tf_vectorizer.get_feature_names_out()
print_top_words(lda, tf_feature_names, 100)
print("Topics Obtained in %0.3fs." % (time() - t0), "\n")

# Print the First 10 Words of the Dictionary
print("Print the First 10 Words of the Dictionary", "\n")
print(tf_feature_names[0:10], "\n")

# Dataframe Word and Topic Association and Saving in CSV

dict_to_list = tf_vectorizer.get_feature_names()
twm_to_list = topic_word_matrix.ravel().tolist()
word_assoc_df = pd.DataFrame({'Word': dict_to_list,
                              'Topic Association': twm_to_list})
print("Before Sorting:")
print(word_assoc_df)
sorted_word_assoc_df = word_assoc_df.sort_values(by = ['Topic Association'],
                                                 ascending = False)
print("After Sorting:")
print(sorted_word_assoc_df)
print("Saving CSV")
sorted_word_assoc_df.to_csv('df_word_assoc.csv', index=False)
print("CSV Saved")

print("Code Executed")
