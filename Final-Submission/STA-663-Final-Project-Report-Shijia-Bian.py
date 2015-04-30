
# coding: utf-8

# Bayesian Hierarchical Clustering 
# =============
# 
# #### STA 663 Computational Statistics in Python Final Project
# 
# #### Shijia Bian
# 
# The Final Project Git Directory: https://github.com/shijbian/STA-663-Final-Project-Submission  
# 
# The Entire Project Draft Tracking Git Directory: https://github.com/shijbian/STA-663-Final-Project

# ### Abstract
# 

# ### Outline  
# 
# + #### Background  
# 
# + #### Traditional Hierarchical Clustering  
# 
#     + Algorithm
#     + Four Main Types of Linkage
#     + Example of Traditional Hierarchical Clustering Model
# 
# + #### BHC Algorithm Debrief  
# 
#     + Diagram
#     + Notation
#     + Hypothesis Testing
#     + Marginal Likelihood for the Hypothesis
#     + Posterial Likelihood for the Hypothesis
#     + Pseudocode for General Implementation
# 
# + #### Case Study: One Dimensional Gaussian Distribution with Unknown Mean and Known Variance
#     + Case Study on BHC Model;
#         + Step 1: Data Simulation
#         + Step 2: Data Visualization
#         + Step 3: Initialization
#         + Step 4: Implement the Function for the Integration of the Likelihood: mu_int
#         + Step 5: The Main Function: hcluster
#         + Step 6: Run the Main Function: hcluster
#         + Step 7: Algorithm Performance and Code Test
#         
#     + Case Study on Traditional Hierarchical Model.
# 
# + #### Limitation of Implemented Algorithm
# 
# + #### Further Improvement and Explanation of the Difficulty
# 
# + #### Conclusion 
# 
# + #### Citation
# 

# ### Background
# 
# This final project is largly based on the paper *Bayesian Hierarchical Clustering* by Katherine A. Heller and Zoubin Ghahramani. This paper is to introduce the new Agglomerative Hierarchical Clustering Algorithm from the perspective of Bayesian. This project is to dig into the paper and explain the paper in layman's term. The main part of this project is to give a specific example by applying the algorithm in the paper on this topic. 
# 
# Agglomerative Hierarchical Algorithm is an important method in unsupervised learning: a useful technique for visualizing and discovering groups in a data set by analyzing the data to be a nested tree dendrogram with nodes. Both the traditional hierarchical model and the Bayesian Hierarchical Clustering(BHC) are Agglomerative Hierarchical trees. Agglomerative Hierarchical tree are grown from the bottom to up. The clusters that are most similar to each other will fused to be a branches. The branches will become a new clusters that will continue to be fused with other most similar branches, so there will be new clusters being fused. The steps will continue till all the clusters are merged to become a single tree. Similar to the traditional hierarchical clustering methods, this new method starts each data point in its own cluster and iteratively merges pair of clusters. The BHC has advantages overcomming many issues that cannot be handled properly through traditional Agglomerative Hierarchical Algorithm methods. The BHC applies hypothesis testing to decide which data should be merged together or not. Different from the traditional hierarchical model, the BHC will return the probability that two nodes can be merged together. 
# 
# In this final project, we will look deep into both the traditional hierarchical model and BHC by providing examples.

# ### Traditional Hierarchical Clustering  
# 
# #### Algorithm (from *An Intro to Statistical Learning*)
# 
# + Start with n observations. Consider each of the observation is a cluster, therefore, we have n clusters. Apply a measure, commonly Euclidean distance, of all the $\binom{n}{2}=n(n-1)/2$ to measure pairwise dissimilarities.
# 
# + For $i = 1, \dots, n$  
#   + (a) Examine all pairs, and fuse the pair that has the most similarity among the i clusters. The measurements for the cluster similarity is commonly meansured by the four types of Linkage that is listed below.
#   + (b) Iterate on conputing the inter-cluster similarity among the $i-1$ remaining clusters. 
#   
# #### Four Main Types of Linkage
# + Complete: maximal intercluster dissimilarity;
# + Single: minimal intercluster dissimilarity;
# + Average: mean intercluster dissimilarity;
# + Centroid: dissimilarity between the centroid for cluster A and the centroid for cluster B.
# 
# #### Example of Traditional Hierarchical Clustering Model
# 
# Below is an example of the traditional hierarchical clustering model. Here are the data set we use for this example:
# 
# | x_1 | x_2 |
# |-----|-----|
# | 0   | 0   |
# |   1 | 0   |
# | 0   | 1   |
# | 1   | 1   |
# | 0.5 | 0   |
# | 0   | 0.5 |
# | 0.5 | 0.5 |
# | 2   | 2   |
# | 2   | 3   |
# | 3   | 2   |
# | 3   | 3   |
# 
# 
# The first plot is a hierarchical clustering dendrogram by complete linkage. The second plot is a 2-dimensional visualization of the same data that we use for the dendrogram. We can see that the points 5 and 6 are closed to each other by visualizaed distance in the "2-Dimensional Visualization of the Data", and the 2 points are also fused together according to the dendrogram. Here are few points that we need pay attention to the tradistional hiercarchical clustering:  
# 
# + This is a hierarchical model we use complete linkage. Different linkage methods can gennerate different types of hierarchical clustering dendrogram, complete linkage is generally preferred in tems of interpretation;
# + The points 0 and 2 that are linked together in the dendrogram are not closed to each other in terms of euclidean distance. Actually, points 0 and 4 are more closed to each other by visualization the 2-dimensional plots. This is because 0 and 2 are closed in terms of horizontal distance that is almost 0;
# + The number of clustering is not unique. In the hierarchical clustering dendrogram, there are 2 clusters if we use the black dashed line to slice the original cluster; We have four clusters if we use the blue line to slice the cluster; We can even get eight clusters by using the red dashed line. 
# 
# 

# In[1]:

import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram


# In[22]:

# Here is the prepared dataset
# Only this dataset is from stack overflow, the code is written by myself

data_array = np.array([[0,   0  ],
              [1,   0  ],
              [0,   1  ],
              [1,   1  ], 
              [0.5, 0  ],
              [0,   0.5],
              [0.5, 0.5],
              [2,   2  ],
              [2,   3  ],
              [3,   2  ],
              [3,   3  ]])

plt.title('Dendrogram of Traditional Hierarchical Clustering')
z = linkage(data_array, method='complete')
d = dendrogram(z)

plt.axhline(y=2.,color='k',ls='dashed')
plt.axhline(y=1.25,color='b',ls='dashed')
plt.axhline(y=0.8,color='r',ls='dashed')

plt.show()


# In[3]:

figsize(6, 6)
import numpy as np
import matplotlib.pyplot as plt

n = range(0,11)
fig, ax = plt.subplots()
plt.title('2-Dimensional Visualization of the Data')
ax.scatter(data_array[:,0], data_array[:,1])

for i, txt in enumerate(n):
    ax.annotate(txt, (data_array[:,0][i],data_array[:,1][i]))
 


# ### Bayesian Hierarchical Clustering (BHC)
# 
# #### Diagram
# Below is the graphical illustration of how nodes is fused in the Bayesian Hierarchical clustering.
# 
# ![alt text](https://raw.githubusercontent.com/shijbian/STA-663-Final-Project/master/April%2023/nodeGraph.png)
# 
# #### Notation  
# 
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\mathcal{D} = \{\bf{x}^{(1)}, \ldots, \bf{x}^{(n)} \}$: entire data set.
# 
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $T_i$: subtree $i$.  
# 
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\mathcal{D}_i$: data set in subtree $i$.
# 
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $T_i \cup T_j \Rightarrow T_k \rightarrow \mathcal{D}_i \cup \mathcal{D}_j$: tree $T_i$ and tree $T_j$ merge to become a new tree $T_k$. 
# 
# #### Hypothesis Testing
# 
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Null Hypothesis $\mathcal{H}_1^k$**: all data in $\mathcal{D}_k$ are i.i.d generated from the same probabilistic model $P(\bf{x}|\theta)$.  
# 
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Alternative Hypothesis $\mathcal{H}_2^k$**: data in $\mathcal{D}_k$ are from two or more clusters.
# 
# #### Marginal Likelihood for the Hypothesis
# 
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Likelihood for Null Hypothesis:** data at tree $\mathcal{D}_k$ is generated from the same cluster
# 
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  $P(\mathcal{D}_k|H_1^k)= \int P(\mathcal{D}_k|\theta) P(\theta|\beta) \mathcal{d}\theta$
# 
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Likelihood for Alternative Hypothesis:** data at tree $\mathcal{D}_k$ is generated from different cluster
# 
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  $P(\mathcal{D}_k|H_2^k)=P(\mathcal{D}_i|T_i)P(\mathcal{D}_j|T_j)$  
# 
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  **Marginal Probability of the Data in Tree $T_k$:**
# 
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  $P(\mathcal{D}_k|T_k)=\pi_kP(\mathcal{D}_k|H^k_1)+(1-\pi_k)P(\mathcal{D}_k|H^k_2)$  
# 
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  $\pi_k = P(H^k_1)$
# 
# #### Posterial Likelihood for the Hypothesis
# 
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **$r_k=\frac{\pi_kp(\mathcal{D}_k)|\mathcal{H}_1^k}{\pi_kP(\mathcal{D}_k|H^k_1)+(1-\pi_k)P(\mathcal{D}_k|H^k_2)}$**
# 
# #### Pseudocode for General Implementation (*From Heller's Paper*)
# 
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **input** data $\mathcal{D} = {x^{(1)}, x^{(2)}, \ldots, x^{(n)}}$, model $p(x|\theta)$, prior $p(\theta|\beta)$
# 
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **initialize: ** number of clusters $\mathcal{c}=\mathcal{n}$ for $i = 1, \ldots, n$
# 
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **while** $\mathcal{c}>1$ **do**:
# 
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  Find the pair $\mathcal{D}_i$ and $\mathcal{D}_j$ with the highest probability of the merged hypothesis:
#  $$\mathcal{r}_k = \frac{\pi_k p(\mathcal{D}_k|\mathcal{H}_1^k)}{p(\mathcal{D}_k|\mathcal{T}_k)}$$
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Merge $\mathcal{D}_k \leftarrow \mathcal{D}_i \cup \mathcal{D}_j$, $\mathcal{T}_k \leftarrow (\mathcal{T}_i, \mathcal{T}_j)$  
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Delete $\mathcal{D}_i$ and $\mathcal{D}_j$, $c \leftarrow c-1$
# 
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **end while**
# 
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **output: ** Bayesian mixture model where each tree node is a mixture component  
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The tree can be cut at points where $\mathcal{r}_k<0.5$

# ### Case Study: One Dimensional Gaussion Distribution with Unknown Mean and Known Precision
# 
# #### Case Study on Designed BHC Model  
# 
# 
# In this case study, assume that the data are from two normal distributions with unknown mean and known variance.  
# 
# $$X_1 \sim \mathcal{N}(\mu_1,1) \text{ and } X_2 \sim \mathcal{N}(\mu_2,1)$$
# 
# The prior for $\mu_1$ and $\mu_2$ is:  
# 
# $$\mu_1,\mu_2 \sim N(\mu_0,\sigma_0)$$
# 
# There are some assumptions before carrying out the algorithm:
# 
# + The data are restricted from two distinguished normal distribution;
# + the dataset can be normalized through the formula for the standard normal, i.e. it has mean zero and a unit variance;
# + each observation $x^{(i)}$ is independnt and generated from different Gaussian distributions, therefore, the covariance is 0 between the two distribution;
# + the realizations of each variable, $x^{(i)}$ in cluster $\mathcal{D}_j$ are independent and identically distributed and drawn from Gaussian distribution with unknown mean $\mu_j$ and precision $\sigma_j^2$, and the prior on $(\mu_j,\sigma_j^{-2})$ is a normal-gamma distribution with hyperparameter $\mu_0,\sigma_0$.

# 
# 
# $$P(\mathcal{D}|H_1)= \int f(x|\mu)f(\mu) \mathcal{d}\mu = \int_{-\infty}^{\infty} \bigg[\frac{1}{\sigma_0 \sqrt{2\pi}}exp(-\frac{(\mu-\mu_0)^2}{2\sigma_0^2})\bigg]\bigg[(\frac{1}{\sigma \sqrt{2\pi}})^n exp(-\frac{n(\bar{x}_0-\mu)^2}{2\sigma^2})   \bigg]\mathcal{d}\mu$$
# 
# 
# $$P(\mathcal{D}|H_2)= \int f(x|\mu)f(\mu) \mathcal{d}\mu = \int_{-\infty}^{\infty} \bigg[\frac{1}{\sigma_0 \sqrt{2\pi}}exp(-\frac{(\mu-\mu_0)^2}{2\sigma_0^2})\bigg]\bigg[(\frac{1}{\sigma \sqrt{2\pi}})^n exp(-\frac{n(\bar{x}_1-\mu)^2}{2\sigma^2})   \bigg]\mathcal{d}\mu$$
# 

# **GOAL of the CASE STUDY:** In the next step of data simulation, the data that are odd indexed are from the different clusters as the data that are even indexed. Therefore, we want out algorithm to distinguish the odd index from the even index. The out put of the algorithm should return a cluster with 2 sets: one set has only odd number, and another set only has even number.

# First, we need install all the packages we need for this demonstration:

# In[4]:

get_ipython().magic(u'matplotlib inline')
import scipy.stats as stats
from IPython.core.pylabtools import figsize
import numpy as np
figsize(12.5, 4)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

get_ipython().magic(u'matplotlib inline')
import scipy.stats as stats
from IPython.core.pylabtools import figsize
import numpy as np
figsize(12.5, 4)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import itertools

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture

import operator


# + **Step 1: Data Simulation: ** simulate 20 data from two normal distribution: N(0, 1) and N(20, 1). 
# $$X_1 \sim \mathcal{N}(1,1) \text{ and } X_2 \sim \mathcal{N}(20,1)$$
# We call the data set **data**. These two clusters are distinguishable from each other. Our simulation assumes that 50% of the data are from the cluster 0 and 50% are from the cluster 1. 
# 
# The first 5 data are shown below:

# In[5]:

# Mean
np.random.seed(1343)
mean = (1,20)
# Variance and covariance
cov = [[1,0],[0,1]]
n=10
dat = np.random.multivariate_normal(mean,cov,n)
# this is the simulated one-dimensional data set from normal distribution
# normal 
data = dat.reshape(n*2, 1)
data[:5]


# + **Step 2: Data Visualization** Visualize the data set and save the data as data frame in Pandas for easy accessing: assume that we have no idea about the mean of the data set. By abserving the graphical distribution, we think that one cluster is around 0, and another cluster is around 20. Because the data set is large enough, so it is reasonable to have the same prior distirbution for the two clusters;

# In[6]:

plt.hist(data, bins=20, color="k", histtype="stepfilled", alpha=0.8)
plt.title("Histogram of the dataset")
plt.ylim([0, None])
#print data[:10], "..."


# In[7]:

import pandas as pd
# save the data as a data frame in pandas
df = pd.DataFrame({'data': data.reshape(-1)})
df.head()


# + **Step 3: Initialization:**  
# Initialize the data by extracting two points. These two points are from the two clusters respectively. 
# 
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; t_clust_0: initialized $T_i$  
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; t_clust_1: initialized $T_j$   
# 
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Important assumption in this designed algorithm:** we assume that we are able to extract two data points that are from different clusters from the data set.
# 
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; According to the visualization above, we can extract the point 1 and point 2 as initial value. 
# 
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The information we need store are the two points are:   
# 
#    + the index of the point 2: stored into two arrays: init_index_0 and init_index_1;
#    + the corresponding value for the two points: stored into two arrays: clust0_v and clust1_v.
#   

# In[8]:

# The initial index for cluster 0 is stroed into array init_index_0
init_index_0 = np.array([1])

# The initial index for cluster 1 is stroed into array init_index_1
init_index_1 = np.array([2])

# The initial value for cluster 0 is stroed into array clust0_v
clust0_v = np.array([data[1]])

# The initial value for cluster 1 is stroed into array clust1_v
clust1_v = np.array([data[2]])


# + **Step 4: Implement the Function: Computing the Integration of the Liklihood Function：**
# 
# 
# 
# $$P(\mathcal{D}|H_k)= \int f(x|\mu)f(\mu) \mathcal{d}\mu = \int_{-\infty}^{\infty} \bigg[\frac{1}{\sigma_0 \sqrt{2\pi}}exp(-\frac{(\mu-\mu_0)^2}{2\sigma_0^2})\bigg]\bigg[(\frac{1}{\sigma \sqrt{2\pi}})^n exp(-\frac{n(\bar{x}-\mu)^2}{2\sigma^2})   \bigg]\mathcal{d}\mu$$

# In[9]:

from scipy.integrate import quad

def mu_int(sigma_null, mu_null, sigma, N, X):
    k1 = 1.0 / (sigma_null * math.sqrt(2*math.pi))
    s1 = -1.0 / (2 * sigma_null * sigma_null)
    
    k2 = 1.0 / (sigma * math.sqrt(2*math.pi))
    s2 = -1.0 / (2 * sigma * sigma)
    
    x_bar = X*1.0/N
    
    def f(mu):
        return (k1 * math.exp(s1 * (mu - mu_null)*(mu - mu_null)))*(k2 * math.exp(s2 * N*(mu - x_bar)*(mu - x_bar)))
    return f

# The "quad" command below is to run the gaussian distribution 
# integration function:
# quad(mu_int(10, 10, 10, 10, 105), -inf, inf)


# + **Step 5: The Main Function: hcluster：**
#     
# The main function hcluster is designed specifically to accomodate the BHC algorith for one-dimensional data from 2 distinct normal distributions with unknown mean and know variance. 
# 
# In addition, we also have the two points initialized from the two clusters generated from the logic above:
# 
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; t_clust_0: initialized $T_i$  
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; t_clust_1: initialized $T_j$   
# 
# This function follows the description of the algorith in the original paper.  
# 
# +   This function will take into the parameters below:  
# 
#     + df: the one dimensional data set;
#     + mu_null: the prior mean
#     + sigma_null: the prior variance
#     + pi: the probability that select the first cluster;
#     + init_index_0: the index of data that are initialized in the first cluster
#     + init_index_1: the index of data that are initialized in the second cluster
#     + clust0_v: data that are initialized in the first cluster
#     + clust1_v: data that are initialized in the second cluster
# 

# In[10]:

import operator
np.random.seed(134)
def hcluster(df, sigma, mu_null, sigma_null, pi, init_index_0, init_index_1, clust0_v, clust1_v):
    closed = 0.0
    # initialize the T_i for clust 0: 2 initial selected values
    N0 = clust0_v.size
    sumclust0 = sum(clust0_v)
    t_clust_0 = quad(mu_int(sigma_null, mu_null, sigma, N0, sumclust0), -inf, inf)[1]
    
    # initialize the T_j for clust 1: 2 initial selected values
    N1 = clust1_v.size
    sumclust1 = sum(clust1_v)
    t_clust_1 = quad(mu_int(sigma_null, mu_null, sigma, N1, sumclust1), -inf, inf)[1]
    
    n = df.shape[0]
    # go through all the points
    c = 10
    while c > 0:
        for j in range(n):

            closest_lik = 0.0
            clust_0 = dict()
            clust_1 = dict()
            for i in range(n):            
                # traverse of the left array
                # i must not exist in the already exist init_index
                if (i not in init_index_0 and i not in init_index_1):
                    tempx_clust0_v = np.append([clust0_v], [df.ix[i,0]])
                    tempx_clust1_v = np.append([clust1_v], [df.ix[i,0]])
                
                    # likeli for null under clust 0
                    X0 = sum(tempx_clust0_v)
                    N0 = tempx_clust0_v.size
                    lik_alt_0 = pi*quad(mu_int(sigma_null, mu_null, sigma, N0, X0), -inf, inf)[0]
                
                    # likeli for alternative under clust 0
                    X1 = sum(tempx_clust0_v[:N0-1])
                    N1 = tempx_clust1_v.size
                    X2 = sum(tempx_clust1_v)
                    D_i = quad(mu_int(sigma_null, mu_null, sigma, N0-1, X1), -inf, inf)[0]
                    D_j = quad(mu_int(sigma_null, mu_null, sigma, N1, X2), -inf, inf)[0]
                    lik_alt_1 = (1 - pi)*D_i*D_j
                    # the ratio under the 1st cluster
                    r_0 = lik_alt_0/(lik_alt_0+lik_alt_1)
                
                    # likeli for null under cluster 1
                    X0_1 = sum(tempx_clust1_v)
                    N0_1 = tempx_clust1_v.size
                    lik_alt_0_1 = pi*quad(mu_int(sigma_null, mu_null, sigma, N0_1, X0_1), -inf, inf)[0]
           
                    # likeli for alternative under cluster 0
                    X1_1 = sum(tempx_clust1_v[:N0_1-1])
                    N1_1 = tempx_clust0_v.size
                    X2_1 = sum(tempx_clust0_v)
                    D_i_1 = quad(mu_int(sigma_null, mu_null, sigma, N0_1-1, X1_1), -inf, inf)[0]
                    D_j_1 = quad(mu_int(sigma_null, mu_null, sigma, N1_1, X2_1), -inf, inf)[0]
                    lik_alt_1_1 = (1-pi) * D_i_1 * D_j_1
                    # the ratio under the 2nd cluster 
                    r_0_1 = lik_alt_0_1/(lik_alt_0_1+lik_alt_1_1)
                
                    if r_0 < r_0_1:
                        post = lik_alt_0
                        clust_0.update({i:post})
                    else:
                        post1 = lik_alt_0_1
                        clust_1.update({i:post1})
      


            # select the max likelihood from the two lists
            if (bool(clust_0) == True):
                key_0,maxLik_0 = max(clust_0.iteritems(), key=lambda x:x[1])
                clust0_v = np.append(clust0_v,df.ix[key_0,0])
                init_index_0 = np.append(init_index_0,key_0)
            if (bool(clust_1) == True):
                key_1,maxLik_1 = max(clust_1.iteritems(), key=lambda x:x[1])
                clust1_v = np.append(clust1_v,df.ix[key_1,0])
                init_index_1 = np.append(init_index_1,key_1)

            c = c-1
        print init_index_0,init_index_1


# To run the main function, we need first specify some of the parameter:
# 
# + df: this is the proposed dataset the algorithm will take through;
# + sigma = 1, sigma is assumed to be known;
# + mu_null: prior mean. We set this to be 10, because the mean of the data is around 10;
# + sigma_null: prior variance, we set this to be 10, this is a a very diffused prior distribution;
# + pi: the probability to select from the first cluster;
# + init_index_0, init_index_1, clust0_v, clust1_v have been initialized.

# + ** Step 6: Run the Main Function: hcluster：**

# In[11]:

# calculate the mean of the data
np.mean(df,axis=0)


# In[12]:

# run the main function
hcluster(df, sigma = 1, mu_null=10, sigma_null=10, pi=0.5, init_index_0 = init_index_0, init_index_1 = init_index_1, clust0_v = clust0_v, clust1_v = clust1_v)


# The output shows that there are two clusters like below (due to randomness, the output might not be the same everytime):
# 
# $$[1,5,3,9,7,17,11,13,19,15] \text{ and } [ 2,10,18,0,12,16,14,4,8,6]$$  
# 
# According to our set up, the samples with the odd index are from the differnet cluster that the samples with the even index do.  Therefore, our algorithm is really strong in clustering the points to the correct clusters.
# 
# The tree is grown from bottom to the up for each cluster. For the cluster 0, 5 is grouped with the initial point 1, then this new cluster is grouped with 3, etc. Similarly, in the cluster 1, 10 is grouped with the initial point 2 in cluster 10, then the new cluster will be grouped with 18 to become a new cluster, etc. We also can take a look at another example that has 80 data. We can see that the output is also well clustered:
# 

# In[13]:

# try hcluster on 80 data
np.random.seed(1343)
mean = (1,20)
# Variance and covariance
cov = [[1,0],[0,1]]
n=40
dat = np.random.multivariate_normal(mean,cov,n)
# this is the simulated one-dimensional data set from normal distribution

data = dat.reshape(n*2, 1)
data = pd.DataFrame({'data': data.reshape(-1)})

hcluster(data, sigma = 1, mu_null=10, sigma_null=1, pi=0.5, init_index_0 = init_index_0, init_index_1 = init_index_1, clust0_v = clust0_v, clust1_v = clust1_v)


# + ** Step 7: Algorithm Performance and Code Test：**
# 
# The function hcluster successfully distinguished the two clusters in the one-dimensional data set.  
# 
# In addition, we also want to run a py.test. The test below is to test if the value computed by mu_int is a positive number. We writte the mu_int function into a file called integral.py. Then the test is written into a file called test_integral.py. All tests is passed.

# In[14]:

get_ipython().run_cell_magic(u'file', u'integral.py', u'\nfrom scipy.integrate import quad\nimport math\n\ndef mu_int(sigma_null, mu_null, sigma, N, X):\n   k1 = 1.0 / (sigma_null * math.sqrt(2*math.pi))\n   s1 = -1.0 / (2 * sigma_null * sigma_null)\n   \n   k2 = 1.0 / (sigma * math.sqrt(2*math.pi))\n   s2 = -1.0 / (2 * sigma * sigma)\n   \n   x_bar = X*1.0/N\n   \n   def f(mu):\n       return (k1 * math.exp(s1 * (mu - mu_null)*(mu - mu_null)))*(k2 * math.exp(s2 * N*(mu - x_bar)*(mu - x_bar)))\n   return f\n\n# The "quad" command below is to run the gaussian distribution \n# integration function:\n# quad(mu_int(10, 10, 10, 10, 105), -inf, inf)')


# In[15]:

get_ipython().run_cell_magic(u'file', u'test_integral.py', u'\nimport numpy as np\nfrom numpy.testing import assert_almost_equal\nfrom integral import mu_int\nimport numpy.random\n\n\ndef test_non_negativity():\n    for i in range(10):\n        # u, v, w are the values that match the pre-condition of the mu_int\n        u = np.random.uniform(1)\n        v = np.random.uniform(1)\n        w = np.random.uniform(1)\n        N = 3\n        x = np.random.normal(N)\n        assert mu_int(sigma_null = u, mu_null = v, sigma = w, N = N, X = x) >= 0')


# In[16]:

get_ipython().system(u' py.test')


# #### Case Study on Traditional Hierarchical Clustering Model
# 
# Apply the traditional hierarchical clustering model to the data set **data**. We use four methods and draw 4 dendrograms:
# 
# + Complete
# + Single
# + Average
# + Weighted
# 
# We can compare the difference of the four dendrograms from the dendrogram we draw by using the BHC model. According to the dendrograms below, the dendrograms by using single and complete methods are most similar to what we do for the BHC model. Especially, the complete linkage is more distinguishable, because the heights of the clustering change more dramatically than the single linkage. 
# 
# First, we use *Single*:

# In[17]:

# the seed will get the same data as the data we use for the case study
np.random.seed(1343)
mean = (1,20)
# Variance and covariance
cov = [[1,0],[0,1]]
n=10
dat = np.random.multivariate_normal(mean,cov,n)
# this is the simulated one-dimensional data set from normal distribution
# normal 
data = dat.reshape(n*2, 1)

data_dist = pdist(data) # computing the distance
data_link = linkage(data_dist,method='single') # computing the linkage
dendrogram(data_link,labels=data.dtype.names)
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.suptitle('Samples clustering', fontweight='bold', fontsize=14);


# Second, we use *Complete*:

# In[18]:

data_dist = pdist(data) # computing the distance
data_link = linkage(data_dist,method='complete') # computing the linkage
dendrogram(data_link,labels=data.dtype.names)
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.suptitle('Samples clustering', fontweight='bold', fontsize=14);


# Third, we use *average*:

# In[19]:

data_dist = pdist(data) # computing the distance
data_link = linkage(data_dist,method='average') # computing the linkage
dendrogram(data_link,labels=data.dtype.names)
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.suptitle('Samples clustering', fontweight='bold', fontsize=14);


# Fourth, we use *weighted*:

# In[20]:

data_dist = pdist(data) # computing the distance
data_link = linkage(data_dist,method='weighted') # computing the linkage
dendrogram(data_link,labels=data.dtype.names)
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.suptitle('Samples clustering', fontweight='bold', fontsize=14);


# ### Limitation of Implemented Algorithm
# 
# The algorithm in the case study above is for one-dimensinal data from two distinct normal distribution. For further implememntaion:
# 
# + The initialization process of choosing the two initial points need to be more automatic. The only algorithm I can think of currently is the greedy algorithm that choose the 2 initial points pairwise, like the traditional hierarchical clustering algorithm does.
# + When the sample size is larger than 100, there is possible an integration error;
# + make the algorithm to accomodate the multinomial distribution;
# + withough given the exact number of clusters the data are from, the algorithm can learn the number of clusters the data from through the iteraion;
# + Without given the conjugate distribution, the algorithm can find the optimal likelihood through the EM optimization.

# ### Further Improvement and Explanation of the Difficulty

# For the next step, we want to implement a two dimensional clustering model. The function hcluster should be similar to that of the one-dimensional clustering model. The main difficult part is the integration of the marginal likelihood function.  
# 
# Assume that the data is bivariate normal distribution:
# 
# $ x = 
# \begin{bmatrix}
#     X_1\\ 
#     X_2
# \end{bmatrix}
# $
# 
# $ \mu = 
# \begin{bmatrix}
#     \mu_1\\ 
#     \mu_2
# \end{bmatrix}
# $
# 
# $ \Sigma = 
# \begin{bmatrix}
#     \sigma^2_1 & \sigma^2_{X_1 X_2}\\ 
#     \sigma^2_{X_1 X_2} &\sigma^2_2
# \end{bmatrix}
# $
# 
# The prior distribution for $\mu$ is:  
# 
# $ \mu_0 = 
# \begin{bmatrix}
#     \mu_{10}\\ 
#     \mu_{20}
# \end{bmatrix}
# $
# 
# $ \Sigma = 
# \begin{bmatrix}
#     \sigma^2_{10} & \sigma^2_{X_{10} X_{20}}\\ 
#     \sigma^2_{X_{10} X_{20}} &\sigma^2_{20}
# \end{bmatrix}
# $
# 
# 
# Assume that the next considered data is from the first clustering, the marginal likelihood $P(\mathcal{D}|H_1)$ becomes:
# 
# $$P(\mathcal{D}|H_1) = \int f(x|\mu)f(\mu) \mathcal{d}\mu = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} \bigg[ \bigg(\frac{1}{2\pi\sigma_1\sigma_2}\bigg)^n exp \bigg(-\frac{1}{2(1-\rho^2)} \sum_{i=1}^n \bigg[\frac{(x_{1i}-\mu_1)^2}{\sigma_1^2} -\frac{2\rho(x_{1i} - \mu_1)(x_{i2}-\mu_2)}{\sigma_1 \sigma_2}+ \frac{(x_{2i}-\mu_2)^2}{\sigma_2^2} \bigg] \bigg)\bigg]
# \bigg[\bigg(\frac{1}{\sigma_{10} \sigma_{20} \sqrt{2\pi}}\bigg) exp \bigg(\frac{1}{2(1-\rho_0^2)}\bigg[\frac{(\mu_1-\mu_{10})^2}{\sigma_{10}^2} -\frac{2\rho_0(\mu_{1} - \mu_{10})(\mu_2-\mu_{20})}{\sigma_{10} \sigma_{20}}+ \frac{(\mu_{2}-\mu_{20})^2}{\sigma_{20}^2} \bigg] \bigg)   \bigg]\mathcal{d}\mu_1\mathcal{d}\mu_2 $$
# 
# This computation is very complicated to compute: it does not have closed form. Even we know that it can conjugate to become a normal posterior, but we have to consider the normalizer term, otherwise, all the integration will become 1, there is no points to run the iteration on the marginal likelihood.
# 
# But after solve the problem, we can disinguish the two groups in the graph below by BHC:

# In[21]:

# Sample a set of bivariate normal data that are from two distinct clusters
import matplotlib.pyplot as plt

mean = [0,0]
cov = [[1,0],[0,1]] # diagonal covariance, points lie on x or y-axis

mean2 = [20,20]
cov2 = [[1,0],[0,1]] # diagonal covariance, points lie on x or y-axis

x1,y1 = np.random.multivariate_normal(mean,cov,50).T
x2,y2 = np.random.multivariate_normal(mean2,cov2,50).T
x = np.array([x1,x2])
y = np.array([y1,y2])
plt.title('2-dimensonal Data with Two distinguished Clusters')
plt.plot(x,y,'o'); plt.axis('equal'); plt.show()


# ### Conclusion
# 
# Before wrapping up this project, we can see that BHC has more advantages over the traditional algorithm. The traditional Agglomerative Hierarchical Algorithm has limitations. There is no certain rubric to decide the number of clusters to choose: there are generally only four types of linkages to choose from according to similarity measures: complete, single, average and centroid. The “good” or “bad” model can just be decided by intuition. Over-fitting issues are always involved, and there is no proper way to evaluate the degree of over-fitting. Instead, the BHC builds up a probabilistic model by computing marginal probability: the probability that the data merging together. This probabilistic model can be easily applied when evaluate the over-fitting and compute the predictive model. The Bayesian hypothesis testing can be also used to decide to the depth of the dendrogram, this is a more resonable way to than the traditional hierarchical model does. 
# 
# However, the BHC model also has its shortage, for example, the likelihood function does not have the closed form under the integration. Like what we illustrate in the further improvement session. This requires a high computation complexity and delicated derivation. 

# ### Citation

# *Bayesian Hierarchical Clustering*, Katherine A. Heller, Zoubin Ghahramani
# 
# *Bayesian Hierarchical Clustering in Class Slides* http://cs.brown.edu/courses/csci2950-p/fall2011/lectures/2011-10-13_ghosh.pdf
# 
# *Bayesian Hierarchical Clustering for Studying Cancer Gene Expression Data with Unknown Statistics.* Ed. Ferdinando Di Cunto. PLoS ONE 8.10 (2013): e75748. PMC. Web. 30 Apr. 2015.
# 
# *An Introduction to Statistical Learning with Applications in R* Gareth J., Daniela Witten, Trevor H, Robert T.
# 
# *Stackoverflow* http://stats.stackexchange.com/ 

# In[ ]:



