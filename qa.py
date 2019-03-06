# NLP Final Project
# Roy Bastien
# Fall 2018

from __future__ import unicode_literals
import sys
import os
import collections
import operator
import nltk
import numpy
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
# comment out the following after first run
nltk.download('punkt')
nltk.download()
####
import spacy
from collections import Counter
import en_core_web_sm

nlp = en_core_web_sm.load()
interrogative_word = ['which', 'what', 'whose', 'who', 'whom', 'where', 'when', 'how', 'why']
commonwords = ['does','did','the','a','an','is','was','were','are','.','has','had','have']

def main():

    ###################################
    # Uncomment following code to run on all files
    ###################################
    input_file = open(sys.argv[1], "r").read().splitlines()
    directory = input_file.pop(0)
    os.chdir(directory)
    filelist = list()
    for file in os.listdir("."):
        filelist.append(file)
    filelist.sort()

    input_filelist = input_file
    data = []
    question_list = []
    answer_file = ""
    question_file = ""

    for input_file in input_filelist:
        for filename in filelist:
            if filename == input_file + ".answers":
                answer_file = open(filename).read().splitlines()
            if filename == input_file + ".questions":
                question_file = open(filename).read().splitlines()
                for line in question_file:
                    if line[:11] == "QuestionID:":
                        question_list.append(line[12:])
            if filename == input_file + ".story":
                story_file = open(filename).read()
                data = build_data(answer_file, question_file, story_file)
                answer_file = ""
                question_file = ""

    # Map of questionIDs(the key) to a list of answers for that question
        answers = data[0]
    # Map of questionIDs(the key) to a list of the difficulty and the question
        questions = data[1]
    # List of sentences in the story for the first item of the list containing the header, etc.
        sentences = data[2]

        qlist = []
        ori_qlist = []
        # print(questions)

        for key, question in questions.items():
            pos_t_q = pos_tag_sentence(question[0])
            ne_q = ne_sentance(question[0])
            # print pos_t_s
            qlist.append(pos_t_q)
            ori_qlist.append(question[0])

        # chenge question number here
        ne_s_list = []
        for sentence in sentences:
            pos_t_s = pos_tag_sentence(sentence)
            ne_s = ne_sentance(sentence)
            ne_s_list.append(pos_t_s)

        for index, quest in enumerate(question_list):
            pos_t_q = qlist[index]
            test = find_answer_sentence(pos_t_q, ori_qlist[index], ne_s_list, sentences)
            print("QuestionID: " + quest)
            if test > 0:
                print("Answer: " + test)
            else:
                print("Answer: " )
            print('')
        question_list = []


#question is a parsed sentence, text is a list of parsed sentences
#comes from ne_sentance
def find_answer_sentence(question, origin_question, text, sentences):
    rtn = ''
    parsed_question = parseQuestion(question)
    sentence_scores = q_s(parsed_question[1],parsed_question[0], text)
    question_type = parsed_question[0]
    #print (question_type)
    for (i, ss) in enumerate(sentence_scores):
        if rtn != '':
            break
        if i > 8:
            break
        else:
            sentence_index = ss[1]
            origin_sentence = sentences[sentence_index]
            doc = nlp(origin_sentence)
            if question_type == 'Other':
                #print('here')
                rtn = origin_sentence
            if len(doc.ents) == 0:
                rtn = origin_sentence
            elif question_type == 'Money':
                for X in doc.ents:
                    if X.label_ == 'MONEY' and (X.text not in origin_question):
                        rtn = X.text
            elif question_type == 'Quantity':
                for X in doc.ents:
                    if X.label_ == 'QUANTITY' and (X.text not in origin_question):
                        rtn = X.text
            elif question_type == 'People':
                for X in doc.ents:
                    if (X.label_ == 'PERSON' or X.label_ == 'NORP' or X.label_ == 'ORG' ) and (X.text not in origin_question):
                        rtn = X.text
            elif question_type == 'Time':
                for X in doc.ents:
                    if (X.label_ == 'DATE' or X.label_ == 'TIME') and (X.text not in origin_question):
                        rtn = X.text
            elif question_type == 'Location':
                for X in doc.ents:
                    if (X.label_ == 'GPE' or X.label_ == 'LOC' or X.label_ == 'FAC') and (X.text not in origin_question):
                        rtn = X.text
            #print (origin_sentence)
    return rtn

def q_s(trimmed_question, iw, text):
    sentence_scores = []
    for i, sentence in enumerate(text):
        score = 0
        for sen_word in sentence:
            sen_word = sen_word[0]
            for word in trimmed_question:
                word = word[0]
                if word in commonwords:
                    continue
                if (sen_word[1] == 'NN' or sen_word[1] == 'NNP' or sen_word[1] == 'NNS') and (word[1] == 'NN' or word[1] == 'NNP' or word[1] == 'NNS'):
                    if WordNetLemmatizer().lemmatize(sen_word[0]) == WordNetLemmatizer().lemmatize(word[0]):
                        score += 1.5
                if (sen_word[1] == 'VB' or sen_word[1] == 'VBD' or sen_word[1] == 'VBG' or sen_word[1] == 'VBN' or sen_word[1] == 'VBP' or sen_word[1] == 'VBZ') and (word[1] == 'VB' or word[1] == 'VBD' or word[1] == 'VBG' or word[1] == 'VBN' or word[1] == 'VBP' or word[1] == 'VBZ') :
                    if WordNetLemmatizer().lemmatize(sen_word[0]) == WordNetLemmatizer().lemmatize(word[0]):
                        score += 1
        if score > 0:
            sentence_scores.append((score,i))
    return (sorted(sentence_scores, reverse=True))
            #print(sentence)

def question_sentence(question, text, question_word):
    max_score = 0
    bestfit = -1
    n_score = 0
    v_score = 0
    if question_word == 'how':
        n_score = 1.5
        v_score = 1
    if question_word == 'who':
        n_score = 1
        v_score = 1.5
    if question_word == 'what':
        n_score = 1
        v_score = 1.5
    if question_word == 'when':
        n_score = 1
        v_score = 1.5
    if question_word == 'where':
        n_score = 1
        v_score = 1.5
    for i, sentence in enumerate(text):
        score = 0
        for sen_word in sentence:
            sen_word = sen_word[0]
            for word in question:
                word = word[0]
                if (sen_word[1] == 'NN' or sen_word[1] == 'NNP' or sen_word[1] == 'NNS') and (word[1] == 'NN' or word[1] == 'NNP' or word[1] == 'NNS'):
                    if WordNetLemmatizer().lemmatize(sen_word[0]) == WordNetLemmatizer().lemmatize(word[0]):
                        score += 1.5
                if (sen_word[1] == 'VB' or sen_word[1] == 'VBD' or sen_word[1] == 'VBG' or sen_word[1] == 'VBN' or sen_word[1] == 'VBP' or sen_word[1] == 'VBZ') and (word[1] == 'VB' or word[1] == 'VBD' or word[1] == 'VBG' or word[1] == 'VBN' or word[1] == 'VBP' or word[1] == 'VBZ') :
                    if WordNetLemmatizer().lemmatize(sen_word[0]) == WordNetLemmatizer().lemmatize(word[0]):
                        score += 1
        if score > max_score:
            max_score = score
            bestfit = i
    # print(text[bestfit])
    return bestfit


def np_chunking(sentence):
    word_tokens = nltk.tokenize.word_tokenize(sentence)


def ne_sentance(sentence):
    ne_recoged = []
    ne_tokens = nltk.tokenize.word_tokenize(sentence)
    ne_tags = nltk.pos_tag(ne_tokens)
    ne_ner = nltk.ne_chunk(ne_tags)

    # print(ne_ner)
    # for ne in ne_ner:
    #     print(ne)

    return ne_ner


def pos_tag_sentence(sentence):
    pos_tagged = []
    word_tokens = nltk.tokenize.word_tokenize(sentence)

    for token in word_tokens:
        pos_tagged.append(nltk.pos_tag([token]))

    # for tagged in pos_tagged:
    #     print(tagged)

    return pos_tagged


def build_data(answer_file, question_file, story_file):
    answers = build_answers(answer_file)
    questions = build_questions(question_file)
    sentences = build_sentences(story_file)
    return [answers, questions, sentences]


def build_answers(answer_file):
    answers = {}
    question_id = ""

    for line in answer_file:
        if line[:11] == "QuestionID:":
            question_id = line[12:]
        if line[:7] == "Answer:":
            answer_text = line[8:].split(" | ")
            answers[question_id] = answer_text

    # for x, y in answers.items():
    #     print(x, y)

    return answers


def build_questions(question_file):
    questions = {}
    question_id = ""
    question_text = ""

    for line in question_file:
        if line[:11] == "QuestionID:":
            question_id = line[12:]
        if line[:9] == "Question:":
            question_text = line[10:]
        if line[:11] == "Difficulty:":
            difficulty = line[12:]
            questions[question_id] = [question_text, difficulty]

    questions = collections.OrderedDict(sorted(questions.items()))

    # for x, y in questions.items():
    #     print(x, y)

    return questions


def build_sentences(story_file):
    header = story_file.split("\nTEXT:\n")[0].rstrip()

    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    stories = tokenizer.tokenize(story_file.split("\nTEXT:\n")[1].lstrip())
    sentences = []
    sentences.insert(0, header)

    for story in stories:
        story = story.replace('\n', ' ')
        sentences.append(story)

    # for sent in sentences:
    #     print(sent)

    return sentences

def parseQuestion(questToken):
    #interrogative word
    iw_location = -1
    for (i, w) in enumerate(questToken):
        lower_word = w[0][0].lower()
        if lower_word in interrogative_word:
            return (getQuestionKeyWord(questToken, i))
    return ("Not_Found", questToken)

def getQuestionKeyWord(questToken, iw_location):
    iw = questToken[iw_location][0][0].lower()
    wordType = ''
    if iw == 'how':
        nextword = questToken[iw_location + 1][0][0]
        if WordNetLemmatizer().lemmatize(nextword) == WordNetLemmatizer().lemmatize('do') or WordNetLemmatizer().lemmatize(nextword) == WordNetLemmatizer().lemmatize('be'):
            wordType = 'Other'
        elif nextword == 'much':
            wordType = 'Money'
        else:
            wordType = 'Quantity'
    elif iw == 'who' or iw == 'whom' or iw =='whose':
        wordType = 'People'
    elif iw == 'when':
        wordType = 'Time'
    elif iw == 'where':
        wordType = 'Location'
    else:
        wordType = 'Other'
    return (wordType, questToken)


if __name__ == '__main__':
    main()
