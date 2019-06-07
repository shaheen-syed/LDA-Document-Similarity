# coding: utf-8

"""

	Created by: Shaheen Syed
	Date: 06 June 2019

	- packages required 
	logging, textract, glob2, spacy, nltk, seaborn, numpy, pandas, matplotlib, gensim, multiprocessing, joblib

	- Install spacy with the following commands:
	pip install -U spacy
	python -m spacy download en

	- install stopwords from nltk
	import nltk
	nltk.download('stopwords')

	- important when converting pdf to plain text
	use chardet==2.3.0 for textract (to convert pdf to plain text). The latest version of chardet does not work properly in some cases where encoding cannot be retrieved
"""

# packages and modules
import logging
import re # use regular expressions
from datetime import datetime
from helper_functions import *
from gensim import corpora, models # for latent dirichlet allocation
import pandas as pd # for dataframes
import matplotlib.pyplot as plt # for plotting
import seaborn as sns # for plotting
import numpy as np # for vectors and arrays

# for parallel processing
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed

# switches (turn on what needs to be executed)
perform_pdf_to_plain = False
perform_tokenization = False
perform_topic_inference = False
plot_topics_in_documents = False
perform_document_similarity = False
plot_document_similarity = False


"""
	select which LDA model to use
	1: LDA model with 16 topics trained on 72,000 publication abstracts of all 50 fisheries journals from 2000-2017, 
		-	Mapping the global network of fisheries science collaboration
		-	https:doi.org/10.1111/faf.12379 (online soon)
	2: LDA model with 25 topics trained on 46,000 full-text publications from 21 fisheries journals from 1990-2016
		-	Narrow lenses for capturing the complexity of fisheries: A topic analysis of fisheries science from 1990 to 2016
		-	https://doi.org/10.1111/faf.12280
"""
lda_type = 1


"""
	internal helper functions
"""
def process_pdf_to_plain(f, i = 1, total = 1):

	"""
		Convert PDF document to plain text

		Parameters
		----------
		f: string
			location of PDF file
		i: int (optional)
			index of list or batch, used to show status process, not necessary
		total: int (optionall)
			total number of pdfs to convert, used to show status process, not necessary.

	"""

	logging.debug('Processing file: {} {}/{}'.format(f, i + 1, total))

	# convert to plain text
	plain_text = pdf_to_plain(f)

	# check if conversion successful
	if plain_text is not None:

		# some pre-processing of the text (conversion from PDF to plain text, especially with two columns and hyphens can be corrected)
		plain_text = full_text_preprocessing(plain_text)

		# remove french part for certain articles (Can. j. Fish. Science)		
		plain_text = re.sub(r'Re\xb4sume\xb4.*?\[Traduit par la Re\xb4daction\]', '', plain_text)
		plain_text = re.sub(r'R\xe9sum\xe9.*?\[Traduit par la R\xe9daction\]', '', plain_text)

		# extract file name from file and remove the .pdf extension
		file_name = f.split(os.sep)[-1][:-4]

		# save plain text of PDF
		save_plain_text(plain_text, file_name, os.path.join('files', 'plain'))


"""
	Script starts here
"""

if __name__ == "__main__":

	# create logging to console
	set_logger()


	# execute if set to True
	if perform_pdf_to_plain:

		"""
			1) Convert PDF file to plain text
		"""

		# read PDF files from folder
		F = read_directory(os.path.join('files', 'pdf'))[70:]

		# define if parallel processing should be on/off
		parallel = False

		if parallel:
			# use parallel processing to speed up processing time
			executor = Parallel(n_jobs = cpu_count(), backend = 'multiprocessing')
			# create tasks so we can execute them in parallel
			tasks = (delayed(process_pdf_to_plain)(f, i, len(F)) for i, f in enumerate(F))
			# execute task
			executor(tasks)

		else:

			# loop over each file and convert to plain text
			for i, f in enumerate(F):

				# call function to process pdf to plain text conversion
				process_pdf_to_plain(f, i, len(F))



	# execute if set to True
	if perform_tokenization:

		"""
			2) Convert plain text to individual word tokens (bag-of-words)
				- tokenization
				- lemmatization (policies -> policy)
				- remove stop words (the, a, of)
				- remove numbers
				- remove single character tokens
				- group Upercase and lowercase (Fisheries = fisheries)
		"""

		# read plain text files from folder
		F = read_directory(os.path.join('files', 'plain'))

		# load spacy so we can do some NLP things
		nlp = setup_spacy()

		# loop over each file, read plain text, and perform tokenization
		for i, f in enumerate(F):

			logging.debug('Processing file: {} {}/{}'.format(f, i + 1, len(F)))

			# read the content of the file
			doc = read_plain_text(f).decode('utf-8')

			# convert to spacy object
			doc = nlp(doc)

			# tokenize
			tokens = word_tokenizer(doc)

			# remove domain specific stop words
			tokens = remove_domain_specific_stopwords(tokens)

			# extract file name from file
			file_name = f.split(os.sep)[-1][:-4]

			# convert tokens from list to a string by joining them and add a new line (this will create a token per line)
			tokens_plain = '\n'.join(tokens)

			# save tokens as txt file
			save_plain_text(tokens_plain, file_name, os.path.join('files', 'tokens'))



	# execute if set to True 
	if perform_topic_inference:

		"""
			3) infer the topic distribution of a document
		"""

		# location of LDA files
		lda_files_location = os.path.join('files', 'lda', 'model{}'.format(lda_type))

		# load LDA dictionary and corpus
		dictionary, corpus = get_dic_corpus(lda_files_location)

		# load lda model
		model = load_lda_model(lda_files_location)

		# read files with tokens
		F = read_directory(os.path.join('files', 'tokens'))

		# get the topic labels (only for convenience)
		topic_labels = [getTopicLabel(i, lda_type) for i in range(0,len(model.get_topics()))]

		# empty list to store topics in
		df = pd.DataFrame(index = topic_labels)

		# loop over each file, convert tokens into list, infer topic distribution
		for i, f in enumerate(F):

			logging.debug('Processing file: {} {}/{}'.format(f, i + 1, len(F)))

			# read tokens and convert to list
			doc = read_plain_text(f).decode('utf-8').split('\n')
		
			# create bag of words from tokens
			bow = model.id2word.doc2bow(doc)

			# infer document-topic distribution
			topics = model.get_document_topics(bow, per_word_topics = False)
			
			# topics as list, take only the second value of each tuple
			topics = [y for x, y in topics]

			# extract file name from file
			file_name = f.split(os.sep)[-1][:-4]

			# add to dataframe
			df[file_name] = pd.Series(topics, index = topic_labels)


		# save location
		topic_save_location = os.path.join('files', 'topics', 'model{}'.format(lda_type))

		# create directory if not exist
		create_directory(topic_save_location)

		# transpose dataframe
		df = df.T

		# save topic distribution as CSV
		df.to_csv(os.path.join(topic_save_location, 'topics.csv'))



	# execute if set to True
	if plot_topics_in_documents:

		"""
			4) plot distribution of topics in documents (num_publications x topics)
		"""

		# location of topic distribution per document
		topic_distribution_location = os.path.join('files', 'topics', 'model{}'.format(lda_type))
		
		# full path name with file
		topic_distribution_file_location = os.path.join(topic_distribution_location, 'topics.csv')

		# read topic distribution as dataframe
		df = pd.read_csv(topic_distribution_file_location, index_col = 0)

		# plot the heatmap
		ax = sns.heatmap(df, cmap = "Blues", annot = True, vmin = 0., vmax = .6, square = True, annot_kws = {"size": 11}, fmt = '.2f', mask= df <= 0.0, linewidths = .5, cbar = False, yticklabels=True)

		# x axis on top
		ax.xaxis.tick_top()
		
		# rotate y ticks
		plt.yticks(rotation=0)
		
		# rotate x ticks
		plt.xticks(rotation=90, ha = 'left')
		
		# get figure from axis 
		fig = ax.get_figure()
		
		# set figure sizes
		fig.set_size_inches(30, 60)
		
		# save figure
		fig.savefig(os.path.join(topic_distribution_location, 'topics.pdf'), bbox_inches='tight')

	# execute if set to True
	if perform_document_similarity:

		"""
			5) Calculate document similarity
		"""

		# location of topic distribution per document
		topic_distribution_location = os.path.join('files', 'topics', 'model{}'.format(lda_type))
		
		# full path name with file
		topic_distribution_file_location = os.path.join(topic_distribution_location, 'topics.csv')

		# read topic distribution as dataframe
		df = pd.read_csv(topic_distribution_file_location, index_col = 0)

		# number of rows in dataframe
		num_rows = df.shape[0]
	
		# array to store the similarity values
		data = np.zeros((num_rows, num_rows))

		# loop over rows in dataframe
		for i in range(0, num_rows):

			# loop over the same rows again so we can compare them
			for j in range(0, num_rows):

				# get values from row i
				row_i = df.iloc[i].values
				# get values from row j
				row_j = df.iloc[j].values

				# calculate hellinger distance (lower scores are more similiar)
				hellinger_distance = calculate_hellinger_distance(row_i, row_j)

				# add to data
				data[i,j] = hellinger_distance

		# create a new dataframe with the hellinger distances. Also add rows and columns as publication names
		df_new = pd.DataFrame(data, columns = df.index.values, index = df.index.values)

		# save distances to CSV
		df_new.to_csv(os.path.join(topic_distribution_location, 'similarities.csv'))



	# execute if set to True
	if plot_document_similarity:

		"""
			plot document similarity (num_publications x num_publications)
			similarity measured by the hellinger distance between two probability distributions, here between two topic distributions
			note that the hellinger distance is symmetrical, so distance between x,y = y,x
			lower values means documents/publications are similiar
		"""

		# location of topic distribution per document
		topic_distribution_location = os.path.join('files', 'topics', 'model{}'.format(lda_type))
		
		# full path name of similarities
		similarities_file_location = os.path.join(topic_distribution_location, 'similarities.csv')

		# read topic distribution as dataframe
		df = pd.read_csv(similarities_file_location, index_col = 0)

		# plot the heatmap, note that cmap has _r, meaning reverse so low values get darker colors
		ax = sns.heatmap(df, cmap = "Blues_r", annot = True, vmin = 0., vmax = .3, square = True, annot_kws = {"size": 11}, fmt = '.2f', mask= df <= 0.0, linewidths = .5, cbar = False, yticklabels=True, xticklabels=True)

		# x axis on top
		ax.xaxis.tick_top()
		
		# rotate y ticks
		plt.yticks(rotation=0)
		
		# rotate x ticks
		plt.xticks(rotation=90, ha = 'left')
		
		# get figure from axis 
		fig = ax.get_figure()
		
		# set figure sizes
		fig.set_size_inches(50, 50)
		
		# save figure
		fig.savefig(os.path.join(topic_distribution_location, 'similarities.pdf'), bbox_inches='tight')



