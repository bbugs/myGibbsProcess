__author__ = 'susanaparis'

# This file contains several functions that are used in the gibbs_output classes.
# It mostly processes output files from the gibbs sampling framework code from Wim.

import numpy as np
import os


class GibbsOutputTrain(object):
    def __init__(self, wdir='/Users/susanaparis/Documents/Belgium/myModels/data/ToyDataWikipedia/gibbs_output_file/train/lda/lda5/plda/en/'):
        """
        Initialize a GibbsOutput object with vmatrix, nmatrix, theta, phi, alpha, beta, K, D, V
        Precondition: In the directory wdir, there are two files .n and .v that correspond to the counts
        of number of topics per document and number of words per topic.
        """
        self.wdir = wdir
        self.nfile_name = ''
        self.nfile_path = ''
        self.vfile_name = ''
        self.vfile_path = ''

        self.n_samples = 0  # number of samples from the gibbs
        self.D = 0  # number of docs
        self.V = 0  # number of unique words
        self.K = 0  # number of topics

        self.alpha = 0.
        self.beta = 0.


        self.vmatrix = np.array([], dtype=int)
        self.nmatrix = np.array([], dtype=int)

        self.load_all()


        # self.theta = mk_theta(self.nmatrix, self.nfile_name)
        # self.phi = mk_phi(self.vmatrix, self.vfile_name)
        # self.D = self.theta.shape[0]
        # self.V = self.phi.shape[1]

    def __str__(self):
        print "GibbsOutputTrain object"
        print "Summary:"
        print "Working directory", self.wdir
        print "Number of documents", self.D
        print "Number of words in vocabulary", self.V
        print "Number of topics", self.K
        print "alpha", self.alpha
        print "beta", self.beta
        print "Number of samples", self.n_samples
        print "vmatrix shape", self.vmatrix.shape
        print "matrix shape", self.nmatrix.shape
        return ""

    def _get_file_names(self):
        file_names = os.listdir(self.wdir)
        self.nfile_name = [f for f in file_names if f.endswith(".n")][0]  # get file that ends with .n
        self.vfile_name = [f for f in file_names if f.endswith(".v")][0]  # get file that ends with .v

        self.nfile_path = self.wdir + self.nfile_name  # path to n file
        self.vfile_path = self.wdir + self.vfile_name  # path to v file
        return

    def _get_param_from_filename(self, fname):
        # fname = 'alpha_0.5_beta_0.01_nr_topics_5_en.n'
        s = fname.split('_')  # ['alpha', '0.5', 'beta', '0.01', 'nr', 'topics', '5', 'en.n']
        self.alpha = float(s[1])
        self.beta = float(s[3])
        self.K = int(s[6])
        return


    def load_all(self):
        """
        Load all files
        """
        self._get_file_names()
        self._get_param_from_filename(self.nfile_name)

        print "loading vmatrix"
        vn_samples, self.vmatrix = read_count_matrix(self.vfile_path)

        print "loading nmatrix"
        nn_samples, self.nmatrix = read_count_matrix(self.nfile_path)
        assert vn_samples == nn_samples
        assert self.vmatrix.shape[0] == self.nmatrix.shape[1]
        self.n_samples = vn_samples
        self.V = self.vmatrix.shape[1]
        self.D = self.nmatrix.shape[0]
        print "done"


class GibbsOutputInfer(object):
    def __init__(self, wdir='/Users/susanaparis/Documents/Belgium/myModels/data/amz_bilingual_dataset/gibbs_output/output/lda100/inference/plda/en/'):
        """
        Given a directory, this class stores the theta and perplexity of the gibbs infer output
        Precondition: In the directory wdir, there are two files .theta and .perplexity
        """
        # Get file names
        self.wdir = wdir
        self.theta_name = ''
        self.theta_path = ''  # full path to file, including name
        self.perplexity_name = ''
        self.perplexity_path = ''  # full path to file, including name

        self.D = 0  # number of docs
        self.K = 0  # number of topics

        self.alpha = 0.
        self.beta = 0.

        self.theta = np.array([], dtype=np.float)
        self.perplexity = 0.
        self.load_all()


    def __str__(self):
        print "GibbsOutputInfer object"
        print "Summary:"
        print "Working directory", self.wdir
        print "Number of documents", self.D
        print "Number of topics", self.K
        print "alpha", self.alpha
        print "beta", self.beta
        print "theta shape", self.theta.shape
        print "perplexity", self.perplexity
        return ""

    def _get_file_names(self):
        file_names = os.listdir(self.wdir)
        self.theta_name = [f for f in file_names if f.endswith(".theta")][0]  # get file that ends with .n
        self.perplexity_name = [f for f in file_names if f.endswith(".perplexity")][0]  # get file that ends with .v
        assert self.theta_name and self.perplexity_name  # check that files are found

        self.theta_path = self.wdir + self.theta_name  # path to n file
        self.perplexity_path = self.wdir + self.perplexity_name  # path to v file
        return

    def _get_param_from_filename(self, fname):
        # fname = 'alpha_0.5_beta_0.01_nr_topics_5_en.n'
        s = fname.split('_')  # ['alpha', '0.5', 'beta', '0.01', 'nr', 'topics', '5', 'en.n']
        self.alpha = float(s[1])
        self.beta = float(s[3])
        self.K = int(s[6])
        return

    def read_theta(self, fpath):
        """
        Parse the .theta file obtained when inferring in a new corpus
        """
        self.theta = np.loadtxt(fpath)
        return

    def load_all(self):
        """
        Load all files
        """
        self._get_file_names()
        print "Reading alpha, beta and K from file name"
        self._get_param_from_filename(self.theta_name)

        print "loading theta"
        self.read_theta(self.theta_path)

        print "loading perplexity"
        self.perplexity = read_perplexity(self.perplexity_path)

        self.D = self.theta.shape[0]
        assert self.theta.shape[1] == self.K  # check theta dimensions
        print "done loading"


def read_count_matrix(fpath):
    """
    Read in either a vmatrix or nmatrix
    return number of samples and count matrix
    """
    with open(fpath, 'r') as f:
        first_line = f.readline()

        # get number of samples from first line
        # samples: 36
        clue = 'samples: '
        s = first_line.find(clue) + len(clue)
        n_samples = int(first_line[s:])

        # get v matrix from the rest of lines
        rest = f.readlines()
        count_matrix = np.array([np.array(map(int, xi.split())) for xi in rest])

        print fpath
        # print "nsamples", n_samples
        # print "shape", count_matrix.shape
    return n_samples, count_matrix




def mk_cond_prob(count_array, smoothing_param, n_samples=1):
    """ (numpy ndarray) -> numpy ndarray
    Input: an array created from the .n file or .v file from the lda topic model
    Output: an array with the conditional probability P(z_k|D_j) or P(w_i|z_k)
    Smoothing_param is alpha in P(z_k|D_j) and beta in P(w_i|z_k)
    """
    print "Making conditional probability matrix"
    ns = count_array  # e.g., size K x V for P(w_i|z_k)
    n_rows, n_columns = ns.shape
    # in the case of P(z_k|D_j): n_rows: # docs , n_columns: # topics (columns)   (Eq.2)
    # in the case of P(w_i|z_k):  n_rows: # topics, n_columns: # words  (Eq. 1)

    # Compute the sums of each row
    row_sums = ns.sum(axis=1) # row_sums.size is number of docs

    # Compute the conditional probabilities with smoothing parameters.
    # For each element in the matrix, ns, add alpha and divide by row_sum + k*alpha

    cond_prob = ns + n_samples * smoothing_param  # numerator

    # compute cond prob per row:
    for i in range(n_rows):
        s = row_sums[i]
        cond_prob[i, :] = cond_prob[i, :] / (s + n_columns * n_samples * smoothing_param)

    return cond_prob

def check_cond_prob(cond_prob, eps=10 ** (-6)):
    """(numpy darray) -> numpy darray
    Check if the rows of a conditionaly probability array add up to 1
    """
    r, c = cond_prob.shape
    # eps = 10 ** (-10)  # eps in numpy is 10**(-16)
    # but I get rounding error of the order of 10**(-14)
    comp = abs(cond_prob.sum(axis=1) - np.ones((1, r))) < np.ones((1, r)) * eps  # comparison
    return comp.sum() == r



def mk_theta_from_nmatrix(nmatrix, nfile_name):
    """
    Using the output .n matrix from the gibbs sampling, calculate the per-document topic proportions, i.e., theta
    """
    alpha, beta, K = get_param_from_filename(nfile_name)
    theta = mk_cond_prob(nmatrix, alpha)
    return theta

def mk_phi(vmatrix, vfile_name):
    """
    Using the output .v matrix from the gibbs sampling, calculate the per-topic word distributions, i.e., phi
    """
    alpha, beta, K = get_param_from_filename(vfile_name)
    phi = mk_cond_prob(vmatrix, beta)
    return phi


def read_perplexity(fpath):
    with open(fpath, 'r') as f:
        perplexity = float(f.readline().strip())
    return perplexity






#fpath = '/Users/susanaparis/Documents/Belgium/myModels/data/amz_bilingual_dataset/gibbs_output/output/bilda100/inference/plda/desc/alpha_0.5_beta_0.01_nr_topics_100_desc.theta'

#K = 100
#theta_infer = parse_theta_infer(fpath, K)

#p = read_perplexity(fpath)
