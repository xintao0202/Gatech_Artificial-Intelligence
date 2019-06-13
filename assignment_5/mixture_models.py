from __future__ import division
import warnings
import numpy as np
import scipy as sp
from matplotlib import image
from random import randint
from scipy.misc import logsumexp
from helper_functions import image_to_matrix, matrix_to_image, \
                             flatten_image_matrix, unflatten_image_matrix, \
                             image_difference

warnings.simplefilter(action="ignore", category=FutureWarning)

def get_distance(pixel, mean):
    #print np.linalg.norm(pixel-mean,axis=2).shape
    #print pixel
    #print mean
    return np.linalg.norm(pixel-mean,axis=2)


def assign_cluster(image_values,new_image_values,means,k):
    clusters = [None for i in range(k)]
    for i in range(k):
        idx=np.where(new_image_values==means[i])
        clusters[i]=image_values[idx]
    return clusters

def k_means_cluster(image_values, k=3, initial_means=None):
    """

    Separate the provided RGB values into
    k separate clusters using the k-means algorithm,
    then return an updated version of the image
    with the original values replaced with
    the corresponding cluster values.

    params:
    image_values = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]]
    k = int
    initial_means = numpy.ndarray[numpy.ndarray[float]] or None

    returns:
    updated_image_values = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]]
    """
    # TODO: finish this function
    #raise NotImplementedError()
    # When no initial cluster means are provided, k_means_cluster() should choose k random points from the data (without replacement) to use as initial cluster mean
    #image data flattern to pixel: https://stackoverflow.com/questions/32838802/numpy-with-python-convert-3d-array-to-2d
    pixels=image_values.transpose(2,0,1).reshape(-1,3)
    #print initial_means
    #print pixels
    new_image_values=np.copy(image_values)
    if initial_means is None:
        rand_data = np.random.randint(len(image_values), size=k)
        initial_means = image_values[rand_data, :]


    means_old=None
    means_new=initial_means.copy()
    #print means_new.shape
    distances = np.zeros([image_values.shape[0],image_values.shape[1], k], dtype=np.float64)
    colors = np.zeros([image_values.shape[0],image_values.shape[1]], dtype=np.float64)

    #print distances.shape, image_values.shape
    #iter=0

    #print iter,means_new,means_old
    while(not np.array_equal(means_old,means_new)):
        # for pixel in pixels:
        #     new_image_values.append(value_min_distance(pixel,means_new)) # calcuate new values based on current means
        #print iter
        for index, mean in enumerate(means_new):
            distances[:,:,index]=get_distance(image_values, mean)
            #print distances
        #determine color of each pixel
        #print iter
        #print distances
        #iter += 1
        colors=np.argmin(distances,axis=2)

        means_old=np.copy(means_new)

        for i in range(k):
            means_new[i]=np.mean(image_values[colors==i],0)


        #print means_old
        #print means_new


    #print new_image_values[0]
    for i in range(k):
            idx=np.where(colors==i)
            #print idx
            new_image_values[idx] = means_new[i]
    #print new_image_values.shape
    return new_image_values

def default_convergence(prev_likelihood, new_likelihood, conv_ctr,
                        conv_ctr_cap=10):
    """
    Default condition for increasing
    convergence counter:
    new likelihood deviates less than 10%
    from previous likelihood.

    params:
    prev_likelihood = float
    new_likelihood = float
    conv_ctr = int
    conv_ctr_cap = int

    returns:
    conv_ctr = int
    converged = boolean
    """
    #print "pre",prev_likelihood
    #print "new", new_likelihood
    increase_convergence_ctr = (abs(prev_likelihood) * 0.9 <
                                abs(new_likelihood) <
                                abs(prev_likelihood) * 1.1)

    if increase_convergence_ctr:
        conv_ctr += 1
    else:
        conv_ctr = 0

    return conv_ctr, conv_ctr > conv_ctr_cap


class GaussianMixtureModel:
    """
    A Gaussian mixture model
    to represent a provided
    grayscale image.
    """

    def __init__(self, image_matrix, num_components, means=None):
        """
        Initialize a Gaussian mixture model.

        params:
        image_matrix = (grayscale) numpy.nparray[numpy.nparray[float]]
        num_components = int
        """
        self.image_matrix = image_matrix
        self.num_components = num_components
        if(means is None):
            self.means = np.zeros(num_components)
        else:
            self.means = means
        self.variances = np.zeros(num_components)
        self.mixing_coefficients = np.zeros(num_components)

    def joint_prob(self, val):
        """Calculate the joint
        log probability of a greyscale
        value within the image.

        params:
        val = float

        returns:
        joint_prob = float
        """
        # TODO: finish this
        #raise NotImplementedError()
        # using eq. 9.12 in the pdf. log P(X)=sum_all_k(mixing_oefficents*gaussian distribution)
        # calculate gaussian distribution GD, coeficients the same
        SumLnGD=sp.misc.logsumexp([np.log(self.mixing_coefficients)-0.5*np.log(2*np.pi*np.asarray(self.variances))-((val-self.means)**2)/(2*np.asarray(self.variances))])
        #LnCo=np.log(self.mixing_coefficients[0]) # all are the same
        return SumLnGD

    def initialize_training(self):
        """
        Initialize the training
        process by setting each
        component mean to a random
        pixel's value (without replacement),
        each component variance to 1, and
        each component mixing coefficient
        to a uniform value
        (e.g. 4 components -> [0.25,0.25,0.25,0.25]).

        NOTE: this should be called before
        train_model() in order for tests
        to execute correctly.
        """
        # TODO: finish this
        #raise NotImplementedError()
        #print self.image_matrix.shape
        for i in range(self.num_components):
            a=np.random.randint(self.image_matrix.shape[0])
            b = np.random.randint(self.image_matrix.shape[1])
            self.means[i] = self.image_matrix[a,b]
            #print self.image_matrix[a,b]
        self.variances = [1.0]* self.num_components
        self.mixing_coefficients = [1.0/ self.num_components]* self.num_components
        # print self.means
        # print self.variances
        # print self.mixing_coefficients

    def train_model(self, convergence_function=default_convergence):
        """
        Train the mixture model
        using the expectation-maximization
        algorithm. Since each Gaussian is
        a combination of mean and variance,
        this will fill self.means and
        self.variances, plus
        self.mixing_coefficients, with
        the values that maximize
        the overall model likelihood.

        params:
        convergence_function = function, returns True if convergence is reached
        """
        # TODO: finish this
        #raise NotImplementedError()
        # Evaluate the responsibilities using the current parameter values
        #print self.joint_prob_all()
        post_likelihood=self.likelihood()
        pre_likelihood=float("-inf")
        #iter=0
        conv_ctr=0
        while(convergence_function(pre_likelihood,post_likelihood,conv_ctr)[1] is not True ):
            #print iter
            pre_likelihood= post_likelihood
            mean_new=[]
            variance_new=[]
            mc_new=[]
            flatten = np.hstack(self.image_matrix)
            for i in range(self.num_components):
                Gamma_Znk=self.joint_prob_all()[:,i]/np.sum(self.joint_prob_all(),axis=1)
                Nk=np.sum(Gamma_Znk)
                mean_new.append(np.sum(Gamma_Znk*flatten)/Nk)
                #print (Gamma_Znk*(flatten-mean_new[i])*(np.transpose(flatten-mean_new[i]))).shape
                variance_new.append(np.sum(Gamma_Znk*(flatten-mean_new[i])*(np.transpose(flatten-mean_new[i])))/Nk)
                #print Nk,variance_new
                mc_new.append(Nk/(self.image_matrix.shape[0]*self.image_matrix.shape[1]))
            self.means=np.asarray(mean_new)
            self.variances=np.asarray(variance_new)
            self.mixing_coefficients=np.asarray( mc_new)

            post_likelihood= self.likelihood()
            conv_ctr= convergence_function(pre_likelihood,post_likelihood,conv_ctr)[0]
            #print  self.variances
            #iter+=1

    def segment(self):
        """
        Using the trained model,
        segment the image matrix into
        the pre-specified number of
        components. Returns the original
        image matrix with the each
        pixel's intensity replaced
        with its max-likelihood
        component mean.

        returns:
        segment = numpy.ndarray[numpy.ndarray[float]]
        """
        # TODO: finish this
        #raise NotImplementedError()
        posterior_prob=[]
        flatten = np.hstack(self.image_matrix)
        for i in range(self.num_components):
            posterior_prob.append( self.joint_prob_all()[:, i] / np.sum(self.joint_prob_all(), axis=1))
        labels=np.argmax(np.asarray(posterior_prob),axis=0)
        #print self.means
        flatten_matrix= map(lambda x: self.means[x],labels)
        return np.reshape(flatten_matrix,self.image_matrix.shape)

    def joint_prob_all(self):
        #return a 2-D array, ool by components and row by pixels. convient for sum purpose
        N_sample = self.image_matrix.shape[0] * self.image_matrix.shape[1]
        flatten = np.hstack(self.image_matrix)
        val_repeated = np.repeat(flatten, self.num_components)  # repleat elements to enable matrix operation
        means_repeat = np.tile(self.means, N_sample)
        variances_repeat = np.tile(self.variances, N_sample).astype(np.float64)
        mc_repeat = np.tile(self.mixing_coefficients, N_sample)
        # print val_repeated.shape,means_repeat.shape,variances_repeat.shape,mc_repeat.shape
        LnGD = np.log(mc_repeat) - 0.5 * np.log(
            2 * np.pi *variances_repeat) - ((val_repeated - means_repeat) ** 2) / (
            2 * variances_repeat)
        # likelihood = sp.misc.logsumexp([np.log(self.mixing_coefficients) - 0.5 * np.log(
        #         2 * np.pi * np.power(self.variances, 2)) - ((flatten.transpose() - self.means) ** 2) / (
        #                                  2 * np.power(self.variances, 2))],keepdims=True)
        sum_exp = np.exp(LnGD).reshape(-1,self.num_components)

        return sum_exp

    def likelihood(self):
        """Assign a log
        likelihood to the trained
        model based on the following
        formula for posterior probability:
        ln(Pr(X | mixing, mean, stdev)) = sum((n=1 to N), ln(sum((k=1 to K),
                                          mixing_k * N(x_n | mean_k,stdev_k))))

        returns:
        log_likelihood = float [0,1]
        """
        # TODO: finish this
        #raise NotImplementedError()
        sum_exp=self.joint_prob_all()
        likelihood_element=np.log(np.sum(sum_exp,axis=1))
        return np.sum(likelihood_element)

    def best_segment(self, iters):
        """Determine the best segmentation
        of the image by repeatedly
        training the model and
        calculating its likelihood.
        Return the segment with the
        highest likelihood.

        params:
        iters = int

        returns:
        segment = numpy.ndarray[numpy.ndarray[float]]
        """
        # finish this
        #raise NotImplementedError()
        likelihood_max=self.likelihood()
        means_max=self.means
        variances_max=self.variances
        mc_max=self.mixing_coefficients
        for i in range(iters):
            self.initialize_training()
            self.train_model()
            if self.likelihood()>likelihood_max:
                likelihood_max=self.likelihood()
                means_max = np.copy(self.means)
                variances_max = np.copy(self.variances)
                mc_max = np.copy(self.mixing_coefficients)
        self.means=means_max
        self.variances=variances_max
        self.mixing_coefficients=mc_max
        print self.means
        print self.variances
        print self.mixing_coefficients
        return self.segment()

class GaussianMixtureModelImproved(GaussianMixtureModel):
    """A Gaussian mixture model
    for a provided grayscale image,
    with improved training
    performance."""

    def __init__(self, image_matrix, num_components, means=None):
        """
        Initialize a Gaussian mixture model.

        params:
        image_matrix = (grayscale) numpy.nparray[numpy.nparray[float]]
        num_components = int
        """
        self.image_matrix = image_matrix
        self.num_components = num_components
        if (means is None):
            self.means = np.zeros(num_components)
        else:
            self.means = means
        self.variances = np.zeros(num_components)
        self.mixing_coefficients = np.zeros(num_components)

    # def get_distance_new(self, value, mean):
    #     # print np.linalg.norm(pixel-mean,axis=2).shape
    #     # print pixel
    #     # print mean
    #     return np.linalg.norm(value - mean, axis=0)

    def k_means(self,image_values, k=3, initial_means=None):
        """

        Separate the provided RGB values into
        k separate clusters using the k-means algorithm,
        then return the means

        params:
        image_values = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]]
        k = int
        initial_means = numpy.ndarray[numpy.ndarray[float]] or None

        returns:
        updated_image_values = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]]
        """
        # TODO: finish this function
        # raise NotImplementedError()
        # When no initial cluster means are provided, k_means_cluster() should choose k random points from the data (without replacement) to use as initial cluster mean
        # image data flattern to pixel: https://stackoverflow.com/questions/32838802/numpy-with-python-convert-3d-array-to-2d

        # print initial_means
        # print pixels
        for i in range(self.num_components):
            a = np.random.randint(self.image_matrix.shape[0])
            b = np.random.randint(self.image_matrix.shape[1])
            self.means[i] = self.image_matrix[a, b]

        means_old = None
        means_new = self.means.copy()
        # print means_new.shape
        distances = np.zeros([image_values.shape[0], image_values.shape[1],k], dtype=np.float64)
        colors = np.zeros([image_values.shape[0], image_values.shape[1]], dtype=np.float64)


        # print distances.shape, image_values.shape
        # iter=0

        # print iter,means_new,means_old
        while (not np.array_equal(means_old, means_new)):
            # for pixel in pixels:
            #     new_image_values.append(value_min_distance(pixel,means_new)) # calcuate new values based on current means
            # print iter
            for index, mean in enumerate(means_new):
                distances[:,:, index] = abs(self.image_matrix-mean)
            #print distances.shape
            # determine color of each pixel
            # print iter
            # print distances
            # iter += 1
            colors = np.argmin(distances, axis=2)
            #print colors.shape


            means_old = np.copy(means_new)

            for i in range(k):
                means_new[i] =  np.mean(image_values[colors == i])


        # print new_image_values.shape
        return means_new

    def initialize_training(self):
        """
        Initialize the training
        process by setting each
        component mean using some algorithm that
        you think might give better means to start with,
        each component variance to 1, and
        each component mixing coefficient
        to a uniform value
        (e.g. 4 components -> [0.25,0.25,0.25,0.25]).
        [You can feel free to modify the variance and mixing coefficient
         initializations too if that works well.]
        """
        # TODO: finish this
        #raise NotImplementedError()

        self.means = self.k_means(self.image_matrix, self.num_components)
        #print self.means
        self.variances = [1.0] * self.num_components
        self.mixing_coefficients = [1.0 / self.num_components] * self.num_components


def new_convergence_function(previous_variables, new_variables, conv_ctr,
                             conv_ctr_cap=10):
    """
    Convergence function
    based on parameters:
    when all variables vary by
    less than 10% from the previous
    iteration's variables, increase
    the convergence counter.

    params:

    previous_variables = [numpy.ndarray[float]]
                         containing [means, variances, mixing_coefficients]
    new_variables = [numpy.ndarray[float]]
                    containing [means, variances, mixing_coefficients]
    conv_ctr = int
    conv_ctr_cap = int

    return:
    conv_ctr = int
    converged = boolean
    """
    # TODO: finish this function
    #raise NotImplementedError()



    increase_convergence_ctr = all(np.abs(previous_variables)*0.95<np.abs(new_variables)) and all(np.abs(new_variables)<np.abs(previous_variables) * 1.05)

    # print all(np.abs(previous_variables) * 0.9 < np.abs(new_variables))
    # print all(np.abs(new_variables) < np.abs(previous_variables) * 1.1)
    # print "inc",increase_convergence_ctr

    if increase_convergence_ctr:
        #print "pre",previous_variables
        #print "post", new_variables
        conv_ctr += 1
    else:
        conv_ctr = 0

    return conv_ctr, conv_ctr > conv_ctr_cap

class GaussianMixtureModelConvergence(GaussianMixtureModel):
    """
    Class to test the
    new convergence function
    in the same GMM model as
    before.
    """

    def train_model(self, convergence_function=new_convergence_function):
        # TODO: finish this function
        #raise NotImplementedError()
        post_vars = np.concatenate((self.means,self.variances,self.mixing_coefficients),axis=0)
        pre_vars = [float("-inf")]*9
        # iter=0
        conv_ctr = 0
        while (convergence_function(pre_vars, post_vars, conv_ctr)[1] is not True):
            # print iter
            pre_vars = post_vars
            mean_new = []
            variance_new = []
            mc_new = []
            flatten = np.hstack(self.image_matrix)
            for i in range(self.num_components):
                Gamma_Znk = self.joint_prob_all()[:, i] / np.sum(self.joint_prob_all(), axis=1)
                Nk = np.sum(Gamma_Znk)
                mean_new.append(np.sum(Gamma_Znk * flatten) / Nk)
                # print (Gamma_Znk*(flatten-mean_new[i])*(np.transpose(flatten-mean_new[i]))).shape
                variance_new.append(
                    np.sum(Gamma_Znk * (flatten - mean_new[i]) * (np.transpose(flatten - mean_new[i]))) / Nk)
                # print Nk,variance_new
                mc_new.append(Nk / (self.image_matrix.shape[0] * self.image_matrix.shape[1]))
            self.means = np.asarray(mean_new)
            self.variances = np.asarray(variance_new)
            self.mixing_coefficients = np.asarray(mc_new)
            post_vars=np.concatenate((self.means,self.variances,self.mixing_coefficients),axis=0)

            conv_ctr = convergence_function(pre_vars, post_vars, conv_ctr)[0]

    def __init__(self, image_matrix, num_components, means=None):
        """
        Initialize a Gaussian mixture model.

        params:
        image_matrix = (grayscale) numpy.nparray[numpy.nparray[float]]
        num_components = int
        """
        self.image_matrix = image_matrix
        self.num_components = num_components
        if (means is None):
            self.means = np.zeros(num_components)
        else:
            self.means = means
        self.variances = np.zeros(num_components)
        self.mixing_coefficients = np.zeros(num_components)

        # def get_distance_new(self, value, mean):
        #     # print np.linalg.norm(pixel-mean,axis=2).shape
        #     # print pixel
        #     # print mean
        #     return np.linalg.norm(value - mean, axis=0)

    def k_means(self, image_values, k=3, initial_means=None):
        """

        Separate the provided RGB values into
        k separate clusters using the k-means algorithm,
        then return the means

        params:
        image_values = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]]
        k = int
        initial_means = numpy.ndarray[numpy.ndarray[float]] or None

        returns:
        updated_image_values = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]]
        """
        # TODO: finish this function
        # raise NotImplementedError()
        # When no initial cluster means are provided, k_means_cluster() should choose k random points from the data (without replacement) to use as initial cluster mean
        # image data flattern to pixel: https://stackoverflow.com/questions/32838802/numpy-with-python-convert-3d-array-to-2d

        # print initial_means
        # print pixels
        for i in range(self.num_components):
            a = np.random.randint(self.image_matrix.shape[0])
            b = np.random.randint(self.image_matrix.shape[1])
            self.means[i] = self.image_matrix[a, b]

        means_old = None
        means_new = self.means.copy()
        # print means_new.shape
        distances = np.zeros([image_values.shape[0], image_values.shape[1], k], dtype=np.float64)
        colors = np.zeros([image_values.shape[0], image_values.shape[1]], dtype=np.float64)

        # print distances.shape, image_values.shape
        # iter=0

        # print iter,means_new,means_old
        while (not np.array_equal(means_old, means_new)):
            # for pixel in pixels:
            #     new_image_values.append(value_min_distance(pixel,means_new)) # calcuate new values based on current means
            # print iter
            for index, mean in enumerate(means_new):
                distances[:, :, index] = abs(self.image_matrix - mean)
            # print distances.shape
            # determine color of each pixel
            # print iter
            # print distances
            # iter += 1
            colors = np.argmin(distances, axis=2)
            # print colors.shape


            means_old = np.copy(means_new)

            for i in range(k):
                means_new[i] = np.mean(image_values[colors == i])

        # print new_image_values.shape
        return means_new

    def initialize_training(self):
        """
        Initialize the training
        process by setting each
        component mean using some algorithm that
        you think might give better means to start with,
        each component variance to 1, and
        each component mixing coefficient
        to a uniform value
        (e.g. 4 components -> [0.25,0.25,0.25,0.25]).
        [You can feel free to modify the variance and mixing coefficient
         initializations too if that works well.]
        """
        # TODO: finish this
        # raise NotImplementedError()
        #print "called"
        self.means = self.k_means(self.image_matrix, self.num_components)
        # print self.means
        self.variances = [1.0] * self.num_components
        self.mixing_coefficients = [1.0 / self.num_components] * self.num_components

def bayes_info_criterion(gmm):
    # TODO: finish this function
    #raise NotImplementedError()

    sample_size=len(np.hstack(gmm.image_matrix))
    #print sample_size
    num_params=3*gmm.num_components
    #print num_params
    likelihood=gmm.likelihood()
    #print likelihood

    return np.log(sample_size) * num_params - 2 * likelihood


def BIC_likelihood_model_test():
    """Test to compare the
    models with the lowest BIC
    and the highest likelihood.

    returns:
    min_BIC_model = GaussianMixtureModel
    max_likelihood_model = GaussianMixtureModel

    for testing purposes:
    comp_means = [
        [0.023529412, 0.1254902],
        [0.023529412, 0.1254902, 0.20392157],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432, 0.59215689],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432, 0.59215689,
         0.71372563],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432, 0.59215689,
         0.71372563, 0.964706]
    ]
    """
    # TODO: finish this method
    #raise NotImplementedError()
    comp_means = [
        [0.023529412, 0.1254902],
        [0.023529412, 0.1254902, 0.20392157],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432, 0.59215689],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432, 0.59215689,
         0.71372563],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432, 0.59215689,
         0.71372563, 0.964706]
    ]

    max_likelihood=float("-inf")
    min_bic=float("inf")
    min_bic_model=None
    max_likelihood_model=None
    image_file = 'images/party_spock.png'
    image_matrix = image_to_matrix(image_file)
    for i in range(6):
        k=i+2
        gmm = GaussianMixtureModel(image_matrix, k, comp_means[i])
        gmm.initialize_training()
        gmm.means = np.copy(comp_means[i])
        gmm.train_model()
        likelihood = gmm.likelihood()
        bic=bayes_info_criterion(gmm)
        #print i, likelihood, bic
        if likelihood>max_likelihood:
            max_likelihood=likelihood
            max_likelihood_model=gmm
        if bic<min_bic:
            min_bic=bic
            min_bic_model=gmm

    return min_bic_model, max_likelihood_model


def BIC_likelihood_question():
    """
    Choose the best number of
    components for each metric
    (min BIC and maximum likelihood).

    returns:
    pairs = dict
    """
    # TODO: fill in bic and likelihood
    #raise NotImplementedError()
    bic = 2
    likelihood = 2
    bic_model, likelihood_model=BIC_likelihood_model_test()
    bic=bic_model.num_components
    likelihood=likelihood_model.num_components
    pairs = {
        'BIC': bic,
        'likelihood': likelihood
    }
    return pairs

def return_your_name():
    # return your name
    # TODO: finish this
    #raise NotImplemented()
    return "Xin Tao"

def bonus(points_array, means_array):
    """
    Return the distance from every point in points_array
    to every point in means_array.

    returns:
    dists = numpy array of float
    """
    # TODO: fill in the bonus function
    # REMOVE THE LINE BELOW IF ATTEMPTING BONUS
    raise NotImplementedError()
    return dists
