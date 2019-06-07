# LDA Document Similarity

Use pre-trained latent Dirichlet allocation (LDA) models to infer document-topic distributions from a set of PDFs. The document-topic distribution provides an overview of what topics are found within each of the PDF documents, and in what proportion (topic 1 = 20%, topic 2 = 5% etc.).

Such document-topic distributions can then be used to calculate the similarity between two PDF documents. Here similarity is calculated with the Hellinger distance, a measure to calculate the distance between two probability distribution. Note that the Hellinger distance is symmetrical, meaning the distance between x and y = distance between y and x; contrary to KL-divergence for instance.

## What to do first

Add PDFs to the folder files/pdf

Packages required:
logging, textract, glob2, spacy, nltk, seaborn, numpy, pandas, matplotlib, gensim, multiprocessing, joblib

Install spacy with the following commands:
```
pip install -U spacy
python -m spacy download en
```

install stopwords from nltk:
```
import nltk
nltk.download('stopwords')
```

important when converting pdf to plain text:
use chardet==2.3.0 for textract (to convert pdf to plain text). The latest version of chardet does not work properly in some cases where encoding cannot be retrieved

## How to run

```
start.py
```

## What switches to turn on (set to True)

There are 6 switches that can be set to True, either all at the same time, or by turning them on one-by-one

* perform_pdf_to_plain
	* convert PDF files to plain text variant
	* creates the files/plain folder

* perform_tokenization
	* convert plain text to individual word tokens (bag-of-words)
		- tokenization
		- lemmatization (policies -> policy)
		- remove stop words (the, a, of)
		- remove numbers
		- remove single character tokens
		- group Upercase and lowercase (Fisheries = fisheries)
	* creates the files/tokens folder

* perform_topic_inference
	* infer the document-topic distribution of a document
	* creates the files/topics folder
	* creates a topics.csv file

* plot_topics_in_documents
	* plot distribution of topics in documents (num_publications x topics)
	* creates topics.pdf

* perform_document_similarity
	* calculate the hellinger distance between document-topic distributions
	* creates similarities.csv

* plot_document_similarity
	* plot document similarity (num_publications x num_publications)

## What topic model to use

There are 2 trained LDA models included in the files/lda folder. Note that these models are trained using corpora from the domain of fisheries science. Infering document-topic distributions will correctly work only if the PDFs can be categorized as 'fisheries'.

* model 1
	* LDA model with 16 topics trained on 72,000 publication abstracts of all 50 fisheries journals from 2000-2017, 
		-	Mapping the global network of fisheries science collaboration
		-	https://doi.org/10.1111/faf.12379 (online soon)
* model 2
	* LDA model with 25 topics trained on 46,000 full-text publications from 21 fisheries journals from 1990-2016
		-	Narrow lenses for capturing the complexity of fisheries: A topic analysis of fisheries science from 1990 to 2016
		-	https://doi.org/10.1111/faf.12280

Set
```
lda_type = 1 or lda_type = 2
```

You can train your own LDA topic model with the code provided here:
https://github.com/shaheen-syed/LDA