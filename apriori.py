# Vishad Pokharel
# vpokhare
# vpokhare@u.rochester.edu
# CSC 240 Project 1

import pandas as pd
import itertools

# class for Item to wrap items properly without loss of the column name
class Item:
    def __init__(self, col_name, value):
        self.col_name = col_name
        self.value = value

df = pd.read_csv('adult.data', header=None) # loading without any headers

# renaming columns from the default 0,1,2,3,4... to something sensible 
# only this part of the entire code is specific to the adult dataset, everything else is generic
column_name = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", 
                    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", 
                        "hours-per-week", "native-country", "income"]
df.columns = column_name

# sampling for improving the efficiency of apriori, set this to false to turn it off
# setting this off runs apriori on the entire dataset
sampling = False
# transaction reduction for improving the efficiency of apriori
transaction_reduce = False

sample_percent = 0.5
sample_size = (int) (sample_percent*len(df))

data = df.sample(n=sample_size) if sampling else df

# setting the support level
support = 0.5

# computing minimum support
minimum_support = int (len(data) * support)

# if sampling, reducing the minimum_support to make sure we reduce the possibility of missing frequent itemset in df
minimum_support = (int (minimum_support*0.95)) if sampling else minimum_support

# list to store the frequent itemsets
frequent_itemset = []

# convert set of frozenset to list of list
def convert_to_list(given_set):
    new_list = []
    for itemset in given_set:
        new_list.append(list(itemset))
    return new_list

# convert list of list to set of frozenset
def convert_to_set(given_list):
    new_set = set()
    for itemset in given_list:
        new_set.add(frozenset(itemset))
    return new_set

# method to return support count of itemsets. itemset must in the the format [x y z] 
# returns no of rows where x y z, together
def get_support_count(itemset):
    # temp_series = pd.Series()
    # for i in range(len(itemset)):
    #     if (i==0):
    #         temp_series = data[data[itemset[i].col_name]==itemset[i].value]
    #     else:
    #         temp_series = pd.merge(temp_series, data[data[itemset[i].col_name]==itemset[i].value], how='inner')
    # return len(temp_series)
    current_df = data
    for item in itemset:
        current_df = current_df[current_df[item.col_name]==item.value]
    return (len(current_df))

# improvement for aprioiri algorithm, transaction reduction
def transaction_reduction(candidate_set):
    # new dataframe
    if (len(candidate_set)==0):
        return data
    temp_df = pd.DataFrame(columns=column_name)
    for itemset in candidate_set:
        new_df = data
        for item in itemset:
            # emulates intersection set operation
            new_df = new_df[new_df[item.col_name]==item.value]
        # emulates union set operation
        temp_df = pd.concat([new_df, temp_df]).drop_duplicates()
    return temp_df

# generates subsets for the next level. ex: passing parent_set with group of item generates set with 2 items,
# passing 2 item produces set with 3 items, and so on.
def generate_subset(parent_set):
    if (len(parent_set)==0):
        return []
    new_set_length = len(parent_set[0])+1
    new_set = set()
    for i in range (len(parent_set)):
        for j in range(i+1, len(parent_set)):
            union = frozenset(parent_set[i]) | frozenset(parent_set[j])
            if (len(union)==new_set_length):
                new_set.add(union)
    return convert_to_list(new_set)

# method returns set containing only those elements that are frequent in regards to the minimum_support
def remove_infrequent_items(given_set):
    if (len(given_set)==0):
        return []
    no_of_elem = len(given_set[0])
    new_set = set()
    for itemset in given_set:
        if (get_support_count(itemset)>=minimum_support):
            new_set.add(frozenset(itemset))
    return convert_to_list(new_set)

# method for pruning stage, returns new set which only contains itemset whose all subsets are frequent
def remove_sets_with_no_subsets_in_prev_set(current_set, prev_set):
    if (len(current_set)==0):
        return []
    length_of_subset = len(prev_set[0])
    new_list = []
    prev_set = convert_to_set(prev_set)
    for current_itemset in current_set:
        current_subsets = list(itertools.combinations(current_itemset, length_of_subset))
        add = True
        for subset in current_subsets:
            if frozenset(subset) in prev_set:
                add = add and True
            else:
                add = add and False
        if add:
            new_list.append(current_itemset)
    return new_list
        
# for each column, creating a series of unique elements with count and adding this series to a list
list_of_series = []
for column in data:
    list_of_series.append(data[column].value_counts())

# this is the list for the first set of the algorithm
first_set = []
# populating the list that meet minimum support requirement
for series in list_of_series:
    for index, value in series.items():
        if (value>=minimum_support):
            # creating temp list since we want the list in certain format to use generate_subset method
            temp_list = []
            temp_list.append(Item(series.name, index))
            first_set.append(temp_list)

# adding the set to the list of frequent itemset
frequent_itemset.append(first_set)

if transaction_reduce: 
    data = transaction_reduction(first_set)

# generate the subset for groups of 2 with the list first_set
second_set_c2 = generate_subset(first_set)

infrequent_removed_l2 = remove_infrequent_items(second_set_c2)

# previous set and current set to keep track of sets
prev_set = []
current_set = infrequent_removed_l2

# adding the set to the list of frequent itemset
frequent_itemset.append(infrequent_removed_l2)

# running the actual algorithm on iteration
while (len(current_set) != 0):
    if transaction_reduce:
        data = transaction_reduction(current_set)
    new_set = generate_subset(current_set)
    new_set = remove_sets_with_no_subsets_in_prev_set(new_set, current_set)
    new_set = remove_infrequent_items(new_set)
    prev_set = current_set
    current_set = new_set
    # if current_set is not empty, add it to the list of frequent itemset
    if (len(current_set)!=0):
        frequent_itemset.append(current_set)
# after the while-loop frequent_itemset contains all the frequent itemsets that we want

if sampling:
    data = df
    new_itemset_list = []
    minimum_support = int (len(data) * support)
    for candidate_set in frequent_itemset:
        new_itemset_list.append(remove_infrequent_items(candidate_set))
    frequent_itemset = new_itemset_list
    print ("SAMPLING IS TURNED ON")

if transaction_reduce:
    print ("TRANSACTION REDUCTION IS TURNED ON")

print ("MINIMUM SUPPORT COUNT: {}".format((int (len(data) * support))))
for candidate_set in frequent_itemset:
    for itemset in candidate_set:
        for item in itemset:
            print ("{}: {}, ".format(item.col_name, item.value), end = "")
        print ("(SUPPORT COUNT: {})".format(get_support_count(itemset)))