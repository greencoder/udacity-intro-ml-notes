#!/usr/bin/python
import os
import re
import pickle
import csv
import pandas
import scipy
from time import time
import string
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn import cross_validation
from sklearn import metrics

def getAddresses(file_name):
    """ given a to/from line of the form To: email1, email2, etc...
        return a string that has isolated the to/from
        probably want to feed right into parseAddresses """
    file_name.seek(0)

    ### often there are multiple lines of "To:" addresses
    line_counter = 1
    in_to_block = False
    in_cc_block = False
    cc_already_found = False

    to_string   = ""
    from_string = ""
    cc_string   = ""
    bcc_string  = ""

    for line in file_name.readlines():
        line_counter += 1
        if line_counter == 4:
            from_string = line

        if line_counter == 5:
            in_to_block = True
            to_string = line
        elif line_counter > 5 and "Subject:" in line:
            in_to_block = False
        elif in_to_block:
            to_string = to_string + line

        if "Cc:" in line and not cc_already_found:
            in_cc_block = True
            cc_string = line
        if "Mime-Version:" in line:
            in_cc_block = False
            cc_already_found = True
        if in_cc_block:
            cc_string = cc_string + line

    return to_string, from_string, cc_string

def parseAddresses(line):
    """ given a to/from line of addresses, parse and put into dict """
    split_line = line.split(":")
    if len(split_line) < 2:
        return
    addresses  = split_line[1][1:] ### strip away "To: " or "From: "
    addresses = ''.join(addresses.split())  ### strip away all whitespace
    addresses = addresses.split(",")
    return addresses

def dictionarizeAddresses(dictionary, addresses):
    for address in addresses:
        if address in dictionary:
            dictionary[address] += 1
        else:
            dictionary[address] = 1

def makeCSV(directory, net_address_counts = False):
    """ for each file in the directory, get all the to and from addresses
        (in other words, get all the addresses in the corpus) and count
        the to and from occurrences of each address
            -- write the to/from counts into to/from_addresses.csv
            -- flag the email as to/from a fraudster and record the file path,
        to_flag and from flag into all_records.csv """

    from_address_dict = {}
    to_address_dict   = {}
    records = []

    record_writer = csv.writer( open("all_records.csv", "wb") )

    for root, dirs, files in os.walk(base_dir):
        for file_name in os.listdir(root):
            if os.path.isfile( os.path.join(root, file_name)):
                file_name = str( root+'/'+file_name )
                print file_name
                f = open(file_name, "r")

                if(net_address_counts):
                    to_string, from_string, cc_string   = getAddresses(f)
                    to_emails   = parseAddresses( to_string )
                    from_emails = parseAddresses( from_string )
                    if to_emails:
                        dictionarizeAddresses(to_address_dict, to_emails)
                    if from_emails:
                        dictionarizeAddresses(from_address_dict, from_emails )

                ### file_path, to_flag, from_flag recorded for each email
                to_flag, from_flag, cc_flag = fraudFlagEmail(f)
                email_record = (file_name, (to_flag or cc_flag), from_flag)
                records.append(email_record)
                record_writer.writerow( email_record )

                f.close()

    ### write out total counts of messages to/from each address
    if net_address_counts:
        to_writer     = csv.writer( open("to_addresses.csv", "wb") )
        from_writer   = csv.writer( open("from_addresses.csv", "wb") )

        for key, value in to_address_dict.items():
            to_writer.writerow([key, value])
        for key, value in from_address_dict.items():
            from_writer.writerow([key, value])

def fraudFlagEmail(f, email_list=None):
    """ given an email file f, and a (manually curated) list of
        emails belonging to fraudsters (fraudster_email_list),
        return a trio of booleans for whether that email is
        to, from, or cc'ing a fraudster """

    f.seek(0)
    fraudster_email_list = []
    if not email_list:
        fraudster_email_list = fraudsterEmails()
    else:
        fraudster_email_list = email_list

    to_string, from_string, cc_string   = getAddresses(f)
    to_emails   = parseAddresses( to_string )
    from_emails = parseAddresses( from_string )
    cc_emails   = parseAddresses( cc_string )

    to_fraudster = False
    from_fraudster = False
    cc_fraudster   = False

    ### there can be many "to" emails, but only one "from", so the
    ### "to" processing needs to be a little more complicated
    if to_emails:
        ctr = 0
        while not to_fraudster and ctr < len(to_emails):
            if to_emails[ctr] in fraudster_email_list:
                to_fraudster = True
            ctr += 1
    if cc_emails:
        ctr = 0
        while not to_fraudster and ctr < len(cc_emails):
            if cc_emails[ctr] in fraudster_email_list:
                cc_fraudster = True
            ctr += 1

    if from_emails:
        for email in from_emails:
            if email in fraudster_email_list:
                from_fraudster = True
    return to_fraudster, from_fraudster, cc_fraudster

def parseOutText(f):
    """ given an email file f, parse out all text below the
        metadata block at the top, stem, remove stopwords,
        and return a string that contains all the words
        in the email (space-separated) """

    f.seek(0)  ### go back to beginning of file (annoying)
    all_text = f.read()
    content = all_text.split("X-FileName:")
    words = ""
    if len(content) > 1:
        text_string = content[1].translate(string.maketrans("", ""), string.punctuation)
        text = text_string.split()
        stemmer = SnowballStemmer("english")
        for word in text:
            stemmed_word = stemmer.stem(word)
#            if stemmed_word not in sw:
            words = words + " " + stemmed_word
#    content = [w for w in words if w.lower() not in sw]
    return words

def benchmark(clf_tuple, X_train, y_train, X_test, y_test):
    ### given a classifier (already all set up) and training/test data,
    ### fit and test the classifier and return some simple data about performance

    def describe(test, naive_guess):
        print "fraud emails:  ", sum(test)
        if naive_guess:
            print "total emails:  ", len(test)
            print "naive guess :  ", float( float(len(test)-sum(test))/len(test) )
        print "-------------"

    print('_' * 80)
    print("Training: ")
    print(clf_tuple[1])
    t0 = time()
    clf = clf_tuple[0]
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)

    print "info on test dataset-----"
#    print y_test
    describe(y_test, True)
    print "info on predictions------"
    describe(pred, False)

    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    precision, recall, score, support = metrics.precision_recall_fscore_support(y_test, pred, average='weighted')
    accuracy = metrics.accuracy_score(y_test, pred)
#    print precision
#    print type(recall)
#    print type(score)
#    print type(support)
    print("precision:  %0.3f" % precision)
    print("recall:     %0.3f" % recall)
    print("f1-score:   %0.3f" % score)
    print("support:    %f"    % support)
    print("accuracy:   %0.3f" % accuracy)

    return score


def fraudEmailStats(make_signal_candidates = False, train_by_to = False):

    print "in fraudEmailStats"
    if not make_signal_candidates and not train_by_to:
        print "fraudEmailStats flags not set to true; won't do anything"

    f = open("data/all_records.csv", "r")
    to_file   = open("data/to_addresses.csv", "r")
    from_file = open("data/from_addresses.csv", "r")

    ### to signal candidates are email addresses that have at least 50 emails to them
    ### from signal candidates have at least 10 emails from them
    ### "both" signal candidates fulfill both "to" and "from"
    if make_signal_candidates:

        to_address_counts = {}
        to_reader = csv.reader(to_file, delimiter=",")
        for row in to_reader:
            address, count = row
            to_address_counts[address] = count

        from_address_counts = {}
        from_reader = csv.reader(from_file, delimiter=",")
        for row in from_reader:
            address, count = row
            from_address_counts[address] = count

        to_signal_candidates = {}
        from_signal_candidates = {}
        both_signal_candidates = {}

        for address in to_address_counts:
            counts = int(to_address_counts[address])
            if counts>50:
                to_signal_candidates[address] = counts
                if address in from_address_counts:
                    from_counts = int(from_address_counts[address])
                    if from_counts>10:
                        both_signal_candidates[address] = (counts, from_counts)
        for address in from_address_counts:
            from_counts = int(from_address_counts[address])
            if from_counts>10:
                from_signal_candidates[address] = from_counts

    ff = open("data/to_signal_candidates.csv", "w")
    for item in to_signal_candidates:
        ff.write(item+","+to_address_counts[item]+"\n")
    ff.close()

    ff = open("data/from_signal_candidates.csv", "w")
    for item in from_signal_candidates:
        ff.write(item+","+from_address_counts[item]+"\n")
    ff.close()

    ff = open("data/both_signal_candidates.csv", "w")
    for item in both_signal_candidates:
        counts = both_signal_candidates[item]
        print item, counts
        ff.write(item+","+str(counts[0])+","+str(counts[1])+"\n")
    ff.close()

    ### treat the "to" and "from" emails as a vocabulary that can
    ### be put into a vectorizer

def trainByTo():

    f = open("data/all_records.csv", "r")
    to_file   = open("data/to_addresses.csv", "r")
    from_file = open("data/from_addresses.csv", "r")

    to_email_matrix = []
    from_flags      = []
    counter = 0
    for line in f:
        counter = counter + 1
        path, to_flag, from_flag = line.split(",")
        email = open(path, "r")
        to_string, from_string, cc_string   = getAddresses(email)
        to_emails   = parseAddresses( to_string )
        from_emails = parseAddresses( from_string )
        cc_emails = parseAddresses( cc_string )
        to_string = ""
        for email in to_emails:
            to_string = to_string + email + " "
        to_email_matrix.append(to_string)

        from_flags.append(from_flag)

        if counter%10000==0:
            print counter

    f.close()
    pickle.dump( to_string, open("data/to_strings_dataset_all.pkl", "w") )
    pickle.dump( to_string, open("data/from_flags_dataset_all.pkl", "w") )
    return to_string, from_flags

def main():

    if make_csv:
        base_dir = "enron_mail_20110402/maildir/"
        makeCSV(base_dir)
    if count_signal_messages:
        base_dir = "enron_mail_20110402/maildir/"
        fraudEmailStats(make_signal_candidates = True)
    if train_by_to:
        base_dir = "enron_mail_20110402/maildir/delainey-d/all_documents/"
        to_matrix, from_flags = trainByTo()

    to_records   = []
    from_records = []
    word_records = []

    if make_records:
        for base_dir in base_dirs:
            for root, dirs, files in os.walk(base_dir):
                for file_name in os.listdir(root):
                    if os.path.isfile( os.path.join(root, file_name)):
                        file_name = str( root+'/'+file_name )
                        print file_name
                        f = open(file_name, "r")
                        to_flag, from_flag, cc_flag = fraudFlagEmail(f)
                        words = parseOutText(f)
                        to_records.append( (to_flag or cc_flag) )
                        from_records.append(from_flag)
                        word_records.append( parseOutText(f) )
                        if flag_fraudster_emails:
                            to_flag, from_flag, cc_flag = fraudFlagEmail(f)
                        f.close()
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(word_records, to_records, test_size=0.25, random_state=42)

        vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                     stop_words='english')
        X_train_transformed = vectorizer.fit_transform(X_train)
        X_test_transformed  = vectorizer.transform(X_test)

        pickle.dump(X_train_transformed, open("X_train_transformed.pkl", "w"))
        pickle.dump(X_test_transformed, open("X_test_transformed.pkl", "w"))
        pickle.dump(y_train, open("y_train.pkl", "w"))
        pickle.dump(y_test, open("y_test.pkl", "w"))

if __name__=="__main__":
    main()