import nltk
import os
import re
import time

TESTING_REVIEW_COUNT = 2018
SYNONYM_UPPER_LIMIT = 2
POSITIVE_REVIEW = 'POS'
NEGATIVE_REVIEW = 'NEG'

lemmatizer = nltk.stem.WordNetLemmatizer()
stemmer = nltk.stem.PorterStemmer()

regex = re.compile('[^a-zA-Z0-9]+')

# stopword = nltk.corpus.stopwords.words('english')
stopword = set([ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "film", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "movie", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ])

def training(train_path):
	_token = {}
	counter = 0

	for filename in os.listdir(train_path):
		try:
			counter+=1
			fd = open(train_path+filename,mode = 'rt', encoding = 'utf8')
			for line in fd:
				word_token = nltk.tokenize.word_tokenize(line)
				for wt in word_token:
					wt = wt.lower()
					if wt not in stopword and regex.match(wt) is None:
						wt = lemmatizer.lemmatize(wt)
						wt = stemmer.stem(wt)
						wt_list = []
						for syn in nltk.corpus.wordnet.synsets(wt):
							lemma_count = 0
							for lemma in syn.lemmas():
								# wt_list.append(lemma.name())
								wt = lemma.name()
								_token[wt]=_token.get(wt,0)+1
								if lemma_count>SYNONYM_UPPER_LIMIT:
									break		
								lemma_count += 1 
						# for wt in wt_list:
						# 	_token[wt]=_token.get(wt,0)+1
		except Exception as e:
			print(e)

	return _token


def testing(test_path, threshold, review_test):
	total_count = 0
	_count = 0
	for filename in os.listdir(test_path):
		review_count = 0
		try:
			fd = open(test_path+filename,mode = 'rt', encoding = 'utf8')
			for review_line in fd:
				review_token = nltk.tokenize.word_tokenize(review_line)
				for rt in review_token:
					if rt not in stopword and regex.match(rt) is None:
						rt = lemmatizer.lemmatize(rt)
						rt = stemmer.stem(rt)
						total_rt = pos_token.get(rt,0)+neg_token.get(rt,0)
						try:
							review_count += (pos_token.get(rt,0)/total_rt-neg_token.get(rt,0)/total_rt)
						except:
							pass
		except Exception as e:
			print(e)

		if review_test == POSITIVE_REVIEW:
			if review_count>threshold:
				_count+=1
		elif review_test == NEGATIVE_REVIEW:
			if review_count<threshold:
				_count+=1
		total_count+=1
		
		if total_count>TESTING_REVIEW_COUNT:
			break

	return _count, total_count


def testing_using_userInput():
	while True:
		review_count = 0
		review = input('Enter review: ')
		review_token = nltk.tokenize.word_tokenize(review)
		for rt in review_token:
			rt = rt.lower()
			print(rt)
			rt = lemmatizer.lemmatize(rt)
			print(rt)
			total_rt = pos_token.get(rt,0)+neg_token.get(rt,0)
			try:
				review_count += (-float(1.0*pos_token.get(rt,0)/total_rt*1.0)+float(1.0*neg_token.get(rt,0)/total_rt*1.0))
			except:
				pass

		print(review_count)
		if review_count>0:
			print('Positive review ;)')
		else:
			print('Negative review :/')

		ch = input('Continue?(y/n) ')
		if ch == 'n':
			break

if __name__ == '__main__':

	start_time = time.time()

	base_path = 'C:/Users/apoov/coursera-test/new_dataset/'
	pos_train_path = base_path+'train/pos/'
	neg_train_path = base_path+'train/neg/'
	pos_test_path = base_path+'test/pos/'
	neg_test_path = base_path+'test/neg/'

	pos_token = {}
	neg_token = {}

	pos_token = training(pos_train_path)
	neg_token = training(neg_train_path)

	total_pos_count = 0
	pos_count = 0

	pos_count, total_pos_count = testing(pos_test_path, 0.001, POSITIVE_REVIEW)
	print("Accuracy with positive test samples is "+str(round(float(1.0*pos_count/total_pos_count*1.0*100),5))+" %")

	total_neg_count = 0
	neg_count = 0

	neg_count, total_neg_count = testing(neg_test_path, -0.001, NEGATIVE_REVIEW)
	print("Accuracy with negative test samples is "+str(round(float(1.0*neg_count/total_neg_count*1.0*100),5))+" %")

	print("Average accuracy for test samples is "+str(round(float(1.0*(pos_count+neg_count)/(total_pos_count+total_neg_count)*1.0*100),5))+" %")

	print("Total Time: " + str(round(time.time()-start_time,3))+" s")