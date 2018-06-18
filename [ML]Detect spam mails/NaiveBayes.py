import operator
from math import log

def get_stats():
  distinct_words = set()
  no_of = {'spam':0 , 'ham':0}
  count_of_words_in_class = {'spam':{}, 'ham':{}}
  total_count_of_words_in_class = {'spam':0 , 'ham':0}
  with open('data/train', 'r') as f:
    read_data = f.readlines()
    for pattern in read_data:
      pattern = pattern.split()
      id = pattern[0]
      pattern.pop(0)
      type = pattern[0]
      pattern.pop(0)
      # Increment the count of 'type' mails
      no_of[type]+=1
      indx = 0
      while indx<len(pattern):
        word = pattern[indx]
        indx += 1
        count = int(pattern[indx])
        indx += 1
        # Add word to a set of distinct words in all mails
        distinct_words.add(word)
        total_count_of_words_in_class[type] += count
				# Increment count of current word in the 'type' mails
        count_of_words_in_class[type][word] = count_of_words_in_class[type].get(word,0) + count

  # Extract the most common words in spam/ham mails
  most_spam = sorted(count_of_words_in_class['spam'].items(), key=operator.itemgetter(1), reverse=True)[:5]
  most_ham = sorted(count_of_words_in_class['ham'].items(), key=operator.itemgetter(1), reverse=True)[:5]
  print('\n',30*'#',sep='')
  print('Most common words in spam mails :')
  for t in most_spam:
  	print(t)
  print('Most common words in ham mails :')
  for t in most_ham:
  	print(t)
  classes = ['spam', 'ham']
  prob = {'spam':{},'ham':{}}
  min_prob={}
  for c in classes:
  	for word in count_of_words_in_class[c]:
  		prob[c][word]=(count_of_words_in_class[c][word]+1)/(len(distinct_words)+total_count_of_words_in_class[c])
  	min_prob[c]=(1.)/(len(distinct_words)+total_count_of_words_in_class[c])
  return [ no_of['spam']/(no_of['spam']+no_of['ham']), 
      no_of['ham']/(no_of['spam'] + no_of['ham']), prob ,min_prob]

def classify_pat(pattern,P_spam,P_ham,P_word_given,min_prob):
  pattern = pattern.split()
  id = pattern[0]
  pattern.pop(0)
  type = pattern[0]
  pattern.pop(0)
  prob = {}
  prob['spam'] = log(P_spam)
  prob['ham'] = log(P_ham)

  for word in pattern:
    prob['spam'] += log(P_word_given['spam'].get(word,min_prob['spam']))
    prob['ham'] += log(P_word_given['ham'].get(word,min_prob['ham']))
  if(prob['spam']>=prob['ham']):
    label='spam'
  else:
    label='ham'
  return type,label

def classify(P_spam,P_ham,P_word_given,min_prob):
  total_tests = 0
  correct_classified = 0
  with open('data/test', 'r') as f:
    read_data = f.readlines()
    total_tests=len(read_data)
    for pattern in read_data:
      type,label=classify_pat(pattern,P_spam,P_ham,P_word_given,min_prob)
      correct_classified += (type==label)
  return correct_classified/total_tests

if __name__== '__main__':

  '''
  	Requirements:
  		- Python3
  		- The data directory is in the same directory that contains the script

    Description:
    	1) (get_stats) Calculate the conditional probabilities of each word for both classes,
    		 Prior probability of each class and apply laplace smoothing using the training data.
    	2) (classify) Load the testing data and call the (classify_pat) method for each mail.
    	3) (classify_pat) Apply the Naive Bayes classification rule and use the summation of the
    		 the log of the probabilites to avoid the value from becoming too small.
  '''

  P_spam,P_ham,P_word_given,min_prob = get_stats()
  print('\n',30*'#',sep='')
  print('Prior Probabilities are:')
  print('P_Spam is: ',P_spam)
  print('P_Ham is: ',P_ham)
  print('\n',30*'#',sep='')
  print('Overall Accuracy is:',
  classify(P_spam,P_ham,P_word_given,min_prob)
  )
  print('\n',30*'#',sep='')
  print('The classifier would be decieved \nif the words used have',
				'higher conditional probabilities \ngiven that the mail is spam than if the mail is ham\n')
