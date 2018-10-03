import nltk
import os
import re
import time


start_time = time.time()
base_path = 'C:/Users/apoov/coursera-test/'
pos_train_path = base_path+'train/pos/'
neg_train_path = base_path+'train/neg/'

pos_test_path = base_path+'test/pos/'
neg_test_path = base_path+'test/neg/'


pos_token = {}
neg_token = {}

lemmatizer = nltk.stem.WordNetLemmatizer()

regex = re.compile('[^a-zA-Z0-9]+')

# stopword = nltk.corpus.stopwords.words('english')
stopword = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "film", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "movie", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]

for filename in os.listdir(pos_train_path):
	try:
		fd = open(pos_train_path+filename,'rt')
		for line in fd:
			word_token = nltk.tokenize.word_tokenize(line)
			for wt in word_token:
				if wt not in stopword and regex.match(wt) is None:
					wt = lemmatizer.lemmatize(wt)
					wt_list = []
					for syn in nltk.corpus.wordnet.synsets(wt):
						for lemma in syn.lemmas():
							wt_list.append(lemma.name())		
					for wt in wt_list:
						pos_token[wt]=pos_token.get(wt,0)+1
	except Exception as e:
		print(e)

for filename in os.listdir(neg_train_path):
	try:
		fd = open(neg_train_path+filename,'rt')
		for line in fd:
			word_token = nltk.tokenize.word_tokenize(line)
			for wt in word_token:
				if wt not in stopword and regex.match(wt) is None:
					wt = lemmatizer.lemmatize(wt)
					wt_list = []
					for syn in nltk.corpus.wordnet.synsets(wt):
						for lemma in syn.lemmas():
							wt_list.append(lemma.name())		
					for wt in wt_list:
						neg_token[wt]=neg_token.get(wt,0)+1
	except Exception as e:
		print(e)


total_pos_count=0
pos_count=0

for filename in os.listdir(pos_test_path):
	review_count = 0
	# review = input('Please enter a movie review: ')
	try:
		fd = open(pos_test_path+filename,'rt')
		for review_line in fd:
			review_token = nltk.tokenize.word_tokenize(review_line)
			for rt in review_token:
				# print(rt)
				total_rt = pos_token.get(rt,0)+neg_token.get(rt,0)
				# print(total_rt, pos_token.get(rt,0), neg_token.get(rt,0))
				try:
					review_count+= (float(1.0*pos_token.get(rt,0)/total_rt*1.0)-float(1.0*neg_token.get(rt,0)/total_rt*1.0))
				except:
					pass
	except Exception as e:
		print(e)

	# print(review_count)

	# if review_count>0.01:
	# 	print('So you liked the movie, Awesome!!')
	# elif review_count<-0.01:
	# 	print("Was it that bad ;(")
	# elif review_count>0:
	# 	print("I'm fairly confused? Was it good??? :|")
	# else:
	# 	print("Pretty Confusing! You didn't liked it :/")

	if review_count>0:
		pos_count+=1
	total_pos_count+=1

print("Accuracy with positive test samples is "+str(round(float(1.0*pos_count/total_pos_count*1.0*100),5))+" %")


total_neg_count=0
neg_count=0

for filename in os.listdir(neg_test_path):
	review_count = 0
	# review = input('Please enter a movie review: ')
	try:
		fd = open(neg_test_path+filename,'rt')
		for review_line in fd:
			review_token = nltk.tokenize.word_tokenize(review_line)
			for rt in review_token:
				# print(rt)
				total_rt = pos_token.get(rt,0)+neg_token.get(rt,0)
				# print(total_rt, pos_token.get(rt,0), neg_token.get(rt,0))
				try:
					review_count+= (float(1.0*pos_token.get(rt,0)/total_rt*1.0)-float(1.0*neg_token.get(rt,0)/total_rt*1.0))
				except:
					pass
	except Exception as e:
		print(e)

	# if review_count>0.01:
	# 	print('So you liked the movie, Awesome!!')
	# elif review_count<-0.01:
	# 	print("Was it that bad ;(")
	# elif review_count>0:
	# 	print("I'm fairly confused? Was it good??? :|")
	# else:
	# 	print("Pretty Confusing! You didn't liked it :/")

	if review_count>0:
		neg_count+=1
	total_neg_count+=1

print("Accuracy with positive test samples is "+str(round(float(1.0*neg_count/total_neg_count*1.0*100),5))+" %")

print("Total Time: " + str(round(time.time()-start_time,3))+" s")


# todo : adding the sysset feature for the words that were most common
# 		 also look for giving correct results