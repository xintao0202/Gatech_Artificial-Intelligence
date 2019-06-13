from __future__ import division

import numpy as np
from math import log
from collections import Counter
import time
import random



class DecisionNode:
    """Class to represent a single node in a decision tree."""

    def __init__(self, left, right, decision_function, class_label=None):
        """Create a decision function to select between left and right nodes.

        Note: In this representation 'True' values for a decision take us to
        the left. This is arbitrary but is important for this assignment.

        Args:
            left (DecisionNode): left child node.
            right (DecisionNode): right child node.
            decision_function (func): function to decide left or right node.
            class_label (int): label for leaf node. Default is None.
        """

        self.left = left
        self.right = right
        self.decision_function = decision_function
        self.class_label = class_label

    def decide(self, feature):
        """Get a child node based on the decision function.

        Args:
            feature (list(int)): vector for feature.

        Return:
            Class label if a leaf node, otherwise a child node.
        """

        if self.class_label is not None:
            return self.class_label

        elif self.decision_function(feature):
            return self.left.decide(feature)

        else:
            return self.right.decide(feature)


def load_csv(data_file_path, class_index=-1):
    """Load csv data in a numpy array.

    Args:
        data_file_path (str): path to data file.
        class_index (int): slice output by index.

    Returns:
        features, classes as numpy arrays if class_index is specified,
            otherwise all as nump array.
    """

    handle = open(data_file_path, 'r')
    contents = handle.read()
    handle.close()
    rows = contents.split('\n')
    out = np.array([[float(i) for i in r.split(',')] for r in rows if r])

    if class_index == -1:
        classes = map(int, out[:, class_index])
        features = out[:, :class_index]
        return features, classes

    elif class_index == 0:
        classes = map(int, out[:, class_index])
        features = out[:, 1:]
        return features, classes

    else:
        return out


def build_decision_tree():
    """Create a decision tree capable of handling the provided data.

    Tree is built fully starting from the root.

    Returns:
        The root node of the decision tree.
    """

    decision_tree_root = None

    # TODO: finish this.
    #raise NotImplemented()
    #totally four attribute, A1, A2, A3, A4
    # if A1==1, then go to left node, wihch is class 1, else check A4,
    decision_tree_root = DecisionNode(None, None, lambda feature: feature[0] == 1)
    decision_tree_root.left = DecisionNode(None, None, None, 1)

    # A4 left and right both check A3
    decision_tree_A4 = DecisionNode(None, None, lambda feature: feature[3]==1)
    decision_tree_A3_left=DecisionNode(None, None, lambda feature: feature[2]==1)
    decision_tree_A3_left.left=DecisionNode(None, None, None, 1)
    decision_tree_A3_left.right=DecisionNode(None, None, None, 0)
    decision_tree_A4.left =decision_tree_A3_left

    decision_tree_A3_right = DecisionNode(None, None, lambda feature: feature[2] == 1)
    decision_tree_A3_right.left = DecisionNode(None, None, None, 0)
    decision_tree_A3_right.right = DecisionNode(None, None, None, 1)
    decision_tree_A4.right =decision_tree_A3_right

    decision_tree_root.right = decision_tree_A4

    return decision_tree_root


def confusion_matrix(classifier_output, true_labels):
    """Create a confusion matrix to measure classifier performance.

    Output will in the format:
        [[true_positive, false_negative],
         [false_positive, true_negative]]

    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.

    Returns:
        A two dimensional array representing the confusion matrix.
    """

    # TODO: finish this.
    #raise NotImplemented()
    matrix=[[0,0],[0,0]]
    for i in range(0,len(true_labels)):
        if true_labels[i]==1:
            if classifier_output[i]==1:
                matrix[0][0]+=1
            else:
                matrix[0][1]+=1
        if true_labels[i]==0:
            if classifier_output[i]==1:
                matrix[1][0]+=1
            else:
                matrix[1][1]+=1
    return matrix

def precision(classifier_output, true_labels):
    """Get the precision of a classifier compared to the correct values.

    Precision is measured as:
        true_positive/ (true_positive + false_positive)

    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.

    Returns:
        The precision of the classifier output.
    """

    # TODO: finish this.
    #raise NotImplemented()
    matrix=confusion_matrix(classifier_output,true_labels)
    tp=matrix[0][0]
    fp=matrix[1][0]
    if tp+fp==0:
        return 0
    else:
        return float(tp/(tp+fp))


def recall(classifier_output, true_labels):
    """Get the recall of a classifier compared to the correct values.

    Recall is measured as:
        true_positive/ (true_positive + false_negative)

    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.

    Returns:
        The recall of the classifier output.
    """

    # TODO: finish this.
    #raise NotImplemented()
    matrix = confusion_matrix(classifier_output, true_labels)
    tp = matrix[0][0]
    fn = matrix[0][1]
    if tp + fn == 0:
        return 0
    else:
        return float(tp / (tp + fn))

def accuracy(classifier_output, true_labels):
    """Get the accuracy of a classifier compared to the correct values.

    Accuracy is measured as:
        correct_classifications / total_number_examples

    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.

    Returns:
        The accuracy of the classifier output.
    """

    # TODO: finish this.
    #raise NotImplemented()
    matrix= confusion_matrix(classifier_output, true_labels)
    tp = matrix[0][0]
    fp = matrix[1][0]
    fn = matrix[0][1]
    tn=matrix[1][1]

    tpn = tp + tn
    total=tp + fp + tn + fn
    if total==0:
        return 0.0
    else:
        return 1.0*tpn/total


def gini_impurity(class_vector):
    """Compute the gini impurity for a list of classes.
    This is a measure of how often a randomly chosen element
    drawn from the class_vector would be incorrectly labeled
    if it was randomly labeled according to the distribution
    of the labels in the class_vector.
    It reaches its minimum at zero when all elements of class_vector
    belong to the same class.

    Args:
        class_vector (list(int)): Vector of classes given as 0 or 1.

    Returns:
        Floating point number representing the gini impurity.
    """
    if len(class_vector) == 0:
        return 1
    #calculat number of zeros
    Zeros=[x for x in class_vector if x == 0]

    prob_of_Zero = 1.0 * len(Zeros) / len(class_vector) #probility of the value is zero
    prob_of_One = 1 - prob_of_Zero #probility that the value is 1
    return 1 - prob_of_Zero * prob_of_Zero - prob_of_One * prob_of_One


def gini_gain(previous_classes, current_classes):
    """Compute the gini impurity gain between the previous and current classes.
    Args:
        previous_classes (list(int)): Vector of classes given as 0 or 1.
        current_classes (list(list(int): A list of lists where each list has
            0 and 1 values).
    Returns:
        Floating point number representing the information gain.
    """
    #raise NotImplemented()

    H_T_a=sum(gini_impurity(c)*float(len(c))/len(previous_classes) for c in current_classes )
    #print gini_impurity(previous_classes),"e"
    return gini_impurity(previous_classes)-H_T_a


class DecisionTree:
    """Class for automatic tree-building and classification."""

    def __init__(self, depth_limit=float('inf')):
        """Create a decision tree with a set depth limit.

        Starts with an empty root.

        Args:
            depth_limit (float): The maximum depth to build the tree.
        """

        self.root = None
        self.depth_limit = depth_limit

    def fit(self, features, classes):
        """Build the tree from root using __build_tree__().

        Args:
            features (list(list(int)): List of features.
            classes (list(int)): Available classes.
        """

        self.root = self.__build_tree__(features, classes)

    def __build_tree__(self, features, classes, depth=0):
        """Build tree that automatically finds the decision functions.

        Args:
            features (list(list(int)): List of features., ndarray
            classes (list(int)): Available classes. list
            depth (int): max depth of tree.  Default is 0.

        Returns:
            Root node of decision tree.
        """
        """ algorithm: 
            1. Check for base cases:
                If all elements of a list are of the same class, return a leaf node with the appropriate class label.
                If a specified depth limit is reached, return a leaf labeled with the most frequent class.
            2. For each attribute alpha: evaluate the normalized gini gain gained by splitting on attribute alpha.
            3. Let alpha_best be the attribute with the highest normalized gini gain.
            4. Create a decision node that splits on alpha_best.
            5. Repeat on the sublists obtained by splitting on alpha_best, and add those nodes as children of this node

        """
        # TODO: finish this.

        classes = map(int, classes)
        element_set = list(set(classes))  # no duplicates in element_set
        # calculate most freqent element in the set that appears in classes
        # frequency = [0] * len(element_set)
        # for i in range(len(element_set)):
        #     for c in classes:
        #         if element_set[i] == c:
        #             # print i
        #             frequency[i] += 1
        # most_frequent = element_set[frequency.index(max(frequency))]

        if classes==[]:
            return DecisionNode(None, None, None, None)
        # If all elements of a list are of the same class, return a leaf node with the appropriate class label.
        # print "after",classes
        if len(element_set)==1:
            return DecisionNode(None, None, None, element_set[0])
        # If a specified depth limit is reached, return a leaf labeled with the most frequent class.
        elif depth == self.depth_limit:
            most_frequent= np.argmax(np.bincount(classes))
            return DecisionNode(None, None, None, most_frequent)
        # For each attribute alpha: evaluate the normalized gini gain gained by splitting on attribute alpha.
        best_gain = float('-inf')
        best_alpha_split_val = None
        best_alpha_index = -1

        # for each alpha
        for alpha_index, alpha in enumerate(features.transpose()):
            if len(set(alpha)) == 1:
                most_frequent= np.argmax(np.bincount(classes))
                return DecisionNode(None, None, None, most_frequent)
            # val = []
            # alpha_sort = sorted(zip(alpha, classes))
            # # print alpha_sort
            # for i in range(1, len(alpha_sort)):
            #     x = alpha_sort[i - 1][0]
            #     y = alpha_sort[i][0]
            #     # print x,y
            #     if alpha_sort[i - 1][1] != alpha_sort[i][1]:
            #         avg = (x + y) / 2.0
            #         val.append(avg)

            # calculate increment for continous values, split based on value in bewteen
            increment = float((max(alpha) - min(alpha)) / 70)
            best_split_value = None
            best_split_gain = float("-inf")
            # for each split value
            #print val
            for split_val in np.arange(min(alpha)+increment, max(alpha)-increment,  increment):
                splitted = [[], []]
                #print split_val,"splitted"
                splitted[0] = [y for x, y in zip(alpha, classes) if x < split_val]  # less than split_val
                splitted[1] = [y for x, y in zip(alpha, classes) if x >= split_val]  # larger or equal to split_val
                #print splitted,"splitted"
                gain = gini_gain(classes, splitted)
                # print split_value_gini_gain
                # check to see if this split is better than before
                if gain > best_split_gain:
                    best_split_gain = gain
                    best_split_value = split_val

            # after loop each split value/threshold, we need to update/check if the split is better than best alpha split
            if best_split_gain > best_gain:
                best_gain = best_split_gain
                best_alpha_split_val = best_split_value
                best_alpha_index = alpha_index



        #print best_alpha_index
        sub_nodes=[]
        best_alpha=features[:, best_alpha_index]
        left_alpha_indices = np.where(best_alpha <= best_alpha_split_val)[0]
        right_alpha_indices = np.where(best_alpha > best_alpha_split_val)[0]

        left_classes = [classes[i] for i in left_alpha_indices]
        right_classes = [classes[i] for i in right_alpha_indices]
        left_features = features[left_alpha_indices]
        right_features = features[right_alpha_indices]

        sub_nodes.append(self.__build_tree__(left_features, left_classes, depth + 1))
        sub_nodes.append(self.__build_tree__(right_features, right_classes, depth + 1))
        return DecisionNode(sub_nodes[0], sub_nodes[1], lambda  a: a[best_alpha_index] < best_alpha_split_val)



    def classify(self, features):
        """Use the fitted tree to classify a list of example features.

        Args:
            features (list(list(int)): List of features.

        Return:
            A list of class labels.
        """

        class_labels = []
        class_labels = [self.root.decide(feature) for feature in features]
        return class_labels



def generate_k_folds(dataset, k):
    """Split dataset into folds.

    Randomly split data into k equal subsets.

    Fold is a tuple (training_set, test_set).
    Set is a tuple (examples, classes).

    Args:
        dataset: dataset to be split.
        k (int): number of subsections to create.

    Returns:
        List of folds.
    """

    # TODO: finish this.

    #raise NotImplemented()
    features=dataset[0]
    sample_size=int(len(features)/k)
    classes=np.asarray(dataset[1]).reshape(len(features),1)

    joined=np.concatenate((features,classes),axis=1)
    np.random.shuffle(joined)
    samples=[]
    for i in range(0,k):
        start_index=int(sample_size*i)
        end_index=int((i+1)*sample_size)

        if end_index>len(features):
           end_index=len(features)-1

        features_train=np.concatenate((joined[:start_index,0:-1],joined[end_index:,0:-1]))
        classes_train = np.concatenate((joined[:start_index, -1], joined[end_index:, -1]))

        features_test=joined[start_index:end_index,0:-1]
        classes_test = joined[start_index:end_index, -1]
        samples.append([(features_train,classes_train),(features_test,classes_test)])
    return samples



class RandomForest:
    """Random forest classification."""

    def __init__(self, num_trees, depth_limit, example_subsample_rate,
                 attr_subsample_rate):
        """Create a random forest.

         Args:
             num_trees (int): fixed number of trees.
             depth_limit (int): max depth limit of tree.
             example_subsample_rate (float): percentage of example samples.
             attr_subsample_rate (float): percentage of attribute samples.
        """

        self.trees = []
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate

    def fit(self, features, classes):
        """Build a random forest of decision trees using Bootstrap Aggregation.

            features (list(list(int)): List of features.
            classes (list(int)): Available classes.
        """

        # TODO: finish this.
        #raise NotImplemented()
        #For every tree we're going to build
        for i in range(self.num_trees):
            #Subsample the examples provided us (with replacement) in accordance with a provided example subsampling rate
            num_rows=range(features.shape[0]) # sample from all rows of features.
            num_attrs=range(features.shape[1])
            num_sampled_rows=int(features.shape[0]*self.example_subsample_rate) # output is sampled rows give a rate
            num_sampled_attrs=int(features.shape[1] * self.attr_subsample_rate)

            sampled_rows=np.random.choice(num_rows,num_sampled_rows).tolist()
            sampled_attrs=np.random.choice(num_attrs,num_sampled_attrs).tolist()
            sample_features = features[sampled_rows][:, sampled_attrs]
            sample_classes = [np.asarray(classes)[i] for i in sampled_rows] # list

            #Fit a decision tree to the subsample of data we've chosen (to a certain depth).
            D_tree=DecisionTree(self.depth_limit)
            D_tree.fit(sample_features,sample_classes)
            self.trees.append((sampled_attrs,D_tree))

    # def most_frequent(self,classes):
    #     classes = map(int, classes)
    #     element_set = list(set(classes))  # no duplicates in element_set
    #     # calculate most freqent element in the set that appears in classes
    #     frequency = [0] * len(element_set)
    #     for i in range(len(element_set)):
    #         for c in classes:
    #             if element_set[i] == c:
    #                 # print i
    #                 frequency[i] += 1
    #     most_frequent = element_set[frequency.index(max(frequency))]
    #     return most_frequent

    def classify(self, features):
        """Classify a list of features based on the trained random forest.

        Args:
            features (list(list(int)): List of features.
        """

        # TODO: finish this.
        #raise NotImplemented()


        labels = []
        for index,tree in self.trees:
            sample_features=features[:,index]
            labels.append(tree.classify(sample_features))
        stack_class = zip(*labels)
        classifiers = map(lambda x: np.argmax(np.bincount(x)), stack_class)
        return classifiers

class ChallengeClassifier:
    """Challenge Classifier used on Challenge Training Data."""

    def __init__(self):
        """Create challenge classifier.

        Initialize whatever parameters you may need here.
        This method will be called without parameters, therefore provide
        defaults.
        """

        # TODO: finish this.
        #raise NotImplemented()
        self.forest = RandomForest(25, 15, 0.3, 0.3)

    def fit(self, features, classes):
        """Build the underlying tree(s).

            Fit your model to the provided features.

        Args:
            features (list(list(int)): List of features.
            classes (list(int)): Available classes.
        """

        # TODO: finish this.
        #raise NotImplemented()
        self.forest.fit(features, classes)

    def classify(self, features):
        """Classify a list of features.

        Classify each feature in features as either 0 or 1.

        Args:
            features (list(list(int)): List of features.

        Returns:
            A list of class labels.
        """

        # TODO: finish this.
        #raise NotImplemented()
        classifier=self.forest.classify(features)
        return classifier


class Vectorization:
    """Vectorization preparation for Assignment 5."""

    def __init__(self):
        pass

    def non_vectorized_loops(self, data):
        """Element wise array arithmetic with loops.

        This function takes one matrix, multiplies by itself and then adds to
        itself.

        Args:
            data: data to be added to array.

        Returns:
            Numpy array of data.
        """

        non_vectorized = np.zeros(data.shape)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                non_vectorized[row][col] = (data[row][col] * data[row][col] +
                                            data[row][col])
        return non_vectorized

    def vectorized_loops(self, data):
        """Element wise array arithmetic using vectorization.

        This function takes one matrix, multiplies by itself and then adds to
        itself.

        Bonnie time to beat: 0.09 seconds.

        Args:
            data: data to be sliced and summed.

        Returns:
            Numpy array of data.
        """

        # TODO: finish this.
        #raise NotImplemented()
        return data+data*data

    def non_vectorized_slice(self, data):
        """Find row with max sum using loops.

        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).

        Args:
            data: data to be added to array.

        Returns:
            Tuple (Max row sum, index of row with max sum)
        """

        max_sum = 0
        max_sum_index = 0
        for row in range(100):
            temp_sum = 0
            for col in range(data.shape[1]):
                temp_sum += data[row][col]

            if temp_sum > max_sum:
                max_sum = temp_sum
                max_sum_index = row
        #print max_sum_index
        return max_sum, max_sum_index

    def vectorized_slice(self, data):
        """Find row with max sum using vectorization.

        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).

        Bonnie time to beat: 0.07 seconds

        Args:
            data: data to be sliced and summed.

        Returns:
            Tuple (Max row sum, index of row with max sum)
        """

        # TODO: finish this.
        #raise NotImplemented()
        subset=data[:100,:]
        max_sum_index = subset.sum(axis=1).argmax()
        #print max_sum_index
        max_sum = subset.sum(axis=1)[ max_sum_index]
        return max_sum, max_sum_index

    def non_vectorized_flatten(self, data):
        """Display occurrences of positive numbers using loops.

         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.

         ie, [(1203,3)] = integer 1203 appeared 3 times in data.

         Args:
            data: data to be added to array.

        Returns:
            List of occurrences [(integer, number of occurrences), ...]
        """

        unique_dict = {}
        flattened = np.hstack(data)
        for item in range(len(flattened)):
            if flattened[item] > 0:
                if flattened[item] in unique_dict:
                    unique_dict[flattened[item]] += 1
                else:
                    unique_dict[flattened[item]] = 1

        return unique_dict.items()

    def vectorized_flatten(self, data):
        """Display occurrences of positive numbers using vectorization.

         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.

         ie, [(1203,3)] = integer 1203 appeared 3 times in data.

         Bonnie time to beat: 15 seconds

         Args:
            data: data to be added to array.

        Returns:
            List of occurrences [(integer, number of occurrences), ...]
        """

        # TODO: finish this.
        #raise NotImplemented()
        #print data[1]
        data_flatten=np.hstack(data)
        #print data_flatten
        positive=data_flatten[np.where(data_flatten>0)]
        unique=np.unique(positive,return_counts=True)
        #print unique[1]
        combined=zip(*unique)
        return combined
        
def return_your_name():
    # return your name
    # TODO: finish this
    #raise NotImplemented()
    return "Xin Tao"