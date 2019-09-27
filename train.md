
## Basic Idea
Mapping of two sets of pre-trained monolingual word embeddings
was first proposed by '__some great guy I don't want to know__'. They uses a small dictionary of '__n__' pairs of words __{__(w<sub>s<sub>1</sub></sub>, w<sub>t<sub>1</sub></sub>), (w<sub>s<sub>2</sub></sub>, w<sub>t<sub>2</sub></sub>),... ,(w<sub>s<sub>n</sub></sub>, w<sub>t<sub>n</sub></sub>)__}__ obtained from Google Translate to learn a transformation matrix __W__ that projects the embeddings vsi of the source language words __w<sub>s<sub>i</sub></sub>__ onto the embeddings __v<sub>t<sub>i</sub></sub>__ of their transaltion words wti in the target language:

$\underset{\operatorname{\textbf{W}}}{\operatorname{min}\mathop{}}\sum\limits_{i=1}^{n} || W_{v_{s_i} - v_{t_i}}||^2$

The trained matrix '__W__' can be used for deteting the translation for any source language word *$w_s$* by simply searching a word in *$w_t$* whose embedding *$v_t$* is the nearest to $W_{v_s}$. Recent work by **someone I don't care** has shown that enforcing the matrix $W$ to be orthogonal can effectively improve the results of mapping.

##  GANs for cross lingual word embeddings
The core of this linear mapping is derived from a dictionary, Soquality and size of the dictionary can be considerably affect the results.
**someone again** has shown that, even without dictionary or any other cross-lingual resources, traning the training the transformation Matrix $W$ is stilll possible using GAN framework. 
#### Standard GAN Framework :
It plays a min-max game between two models:
* Generative model $G$ learns from the source data distribution and tries to generate new samples that appear drawn from the distribution of the target data.
* Discriminator $D$ which discriminates the generated samples from the target data.

Adopting the standard GAN framework here will gonna change the objectives to be:
* Generator $G$ has to learn the transformation matrix $W$ that maps the source language embeddings to the target language embedding space.
* Discriminator $D_l$, (usually a neural network classifier), detects whether the inputs is from the target language

$\underset{G}{\operatorname{min}}\mathop{\underset{D}{\operatorname{max}} \mathop{{   }\mathbb{E}_{v_{t} \sim p_{v_t}}\log{D_t(v_t)}}} + \mathbb{E}_{v_{s} \sim p_{v_s}}\log{(1 - D_l(W_{v_s}))}$



