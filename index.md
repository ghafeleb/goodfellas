## Bias detection using Deep Supervised Contrastive Learning (Goodfellas)
#### [[Project Website]](https://ghafeleb.github.io/goodfellas/)


[Sina Aghaei](mailto:saghaei@usc.edu)<sup>1</sup>, [Zahra Abrishami](mailto:zabrisha@usc.edu)<sup>2</sup>, 
[Ali Ghafelehbash](mailto:ghafeleb@usc.edu)<sup>1</sup>, [Bahareh Harandizadeh](mailto:harandiz@usc.edu)<sup>2</sup>, [Negar Mokhberian](mailto:nmokhber@usc.edu)<sup>2</sup>

<sup>1</sup>Department of Industrial and Systems Engineering, University of Southern California, Los Angeles, CA 90008<br/>
<sup>2</sup>Department of Computer Science, University of Southern California, Los Angeles, CA 90008


## Abstract
In this paper, we propose an end-to-end model to detect ideological bias in news articles. We propose a deep supervised contrastive learning model to learn new representations for text with the goal of separating the embeddings from different classes to expose the bias in the textual data and exploit this bias to identify political orientation of different news articles.

## Introduction

In any nation, media can be a dividing issue as it might reflect topics differently based on the political views or ideological lines. Understanding this implicit bias is becoming more and more critical, specifically by growing numbers of social media and news agencies platforms. Recently the number of research in this domain also is increasing, from detection and mitigation of gender bias[[1]](#1), to polarization detection in political views[[4]](#4) also ideological bias of News[[6]](#6). We think the analysis of bias in the news could be very helpful for the readers as make them more responsible about what they hear or read. 

In this work we want to understand how different are news articles on the same subject but from different political parties (left and right) from each other. We want to detect the potential political bias within news articles. We, as human, can easily identify the different political orientation of articles from opposite parties. For example, how different conservative news agencies such as ''Fox News'' approach a subject like Covid-19 compares to a liberal news agency such as ''CNN''. The question is that can machines detect this political bias as well?
%
A proxy for this goal could be a classifier which tries to classify news articles depending on their political party. Existing approaches such as[[6]](#6) tackle this problem using a classifier on the space of the words embedding. The problem with this approach is that it is not end to end, i.e., the embedding are not trained with the purpose of getting a good classification result. As we can see in figure~\ref{fig:bias-embedding}(right), with general purpose word embedding models such as BERT[[5]](#5), classifying embedded articles might not be straightforward. Having a new representation such as the one shown in figure~\ref{fig:bias-embedding}(left) where it maximizes the distance between embedding from different classes could make the classification task much easier, as in the latent space, the bias is exposed.


<p float="center">
  <a href='https://www.linkpicture.com/view.php?img=LPic5fc59238d28c21286633194'><img src='https://www.linkpicture.com/q/embedding.png' type='image'></a> 
  <a href='https://www.linkpicture.com/view.php?img=LPic5fc59238d28c21286633194'><img src='https://www.linkpicture.com/q/embedding.png' type='image'></a>
</p>
<p align="center">
<b>Figure 1:</b> An ideal latent space (left) where the articles from opposite classes are far from each other which helps to expose the political bias (right) and improves the performance of the classification task.
</p>




To achieve such representation for news articles we propose a modification to the deep contrastive Learning model for unsupervised textual representation introduced in~\cite{giorgi2020declutr}. In~\cite{giorgi2020declutr}, they have a unsupervised contrastive loss which for any given textual segment (aka anchor span) it minimizes the distance between its embedding and the embeddings of other textual segments randomly sampled from nearby in the same document (aka positive spans). It also maximizes the distance between the given anchor from other spans which are not in its neighborhood (aka negative spans). In their model, the positive and negative spans are not chosen according to the label of the documents. We propose to alter their objective to a supervised contrastive loss so that the negative spans are sampled from articles with opposite label. The motivation is to maximize the distance between articles from different classes.



## Problem Formulation
We consider a setting where we have various documents (articles) from two different parties called \emph{liberal} (label being 0) and \emph{conservative} (label being 1). All the documents are about a similar topic, \newsa{Covid-19}.

We sample a batch of <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;N" target="_blank"><img src="https://latex.codecogs.com/png.latex?\small&space;N" title="\small N" /></a> documents from the \emph{liberal} party (class label being 0) and <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;N" target="_blank"><img src="https://latex.codecogs.com/png.latex?\small&space;N" title="\small N" /></a> documents from the \emph{conservative} party (class label being 1). For each document from class <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;k&space;\in&space;\{0,1\}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\small&space;k&space;\in&space;\{0,1\}" title="\small k \in \{0,1\}" /></a> we sample <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;A" target="_blank"><img src="https://latex.codecogs.com/png.latex?\small&space;A" title="\small A" /></a> anchor spans <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;s^k_i,~&space;i&space;\in&space;\{1,\dots,AN\}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\small&space;s^k_i,~&space;i&space;\in&space;\{1,\dots,AN\}" title="\small s^k_i,~ i \in \{1,\dots,AN\}" /></a> and per anchor we sample $P$ positive spans <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;s^k_{i&plus;pAN},~&space;p&space;\in&space;\{1,\dots,P\}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\small&space;s^k_{i&plus;pAN},~&space;p&space;\in&space;\{1,\dots,P\}" title="\small s^k_{i+pAN},~ p \in \{1,\dots,P\}" /></a> following the procedure introduced in~\cite{giorgi2020declutr}.

Given an input span, <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;s^k_i" target="_blank"><img src="https://latex.codecogs.com/png.latex?\small&space;s^k_i" title="\small s^k_i" /></a>, a ''transformer-based language models'' encoder <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;f" target="_blank"><img src="https://latex.codecogs.com/png.latex?\small&space;f" title="\small f" /></a>, maps each token in the input span <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;s^k_i" target="_blank"><img src="https://latex.codecogs.com/png.latex?\small&space;s^k_i" title="\small s^k_i" /></a> to a word embedding.
    
Similar to~\cite{giorgi2020declutr}, a pooler <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;g(.)" target="_blank"><img src="https://latex.codecogs.com/png.latex?\small&space;g(.)" title="\small g(.)" /></a>, maps the encoded anchor spans <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;f(s^k_i)" target="_blank"><img src="https://latex.codecogs.com/png.latex?\small&space;f(s^k_i)" title="\small f(s^k_i)" /></a> to a fixed length embedding <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;g(f(s^k_i))" target="_blank"><img src="https://latex.codecogs.com/png.latex?\small&space;g(f(s^k_i))" title="\small g(f(s^k_i))" /></a>. 
    

We take the average of the positive spans per anchor as follows:

<p align="center">
  <a href="https://www.codecogs.com/eqnedit.php?latex=\huge&space;e^k_{i&plus;AN}&space;=&space;\frac{1}{P}\sum_{p=1}^{P}g(f(s^k_{i&plus;pAN}))" target="_blank"><img src="https://latex.codecogs.com/png.latex?\huge&space;e^k_{i&plus;AN}&space;=&space;\frac{1}{P}\sum_{p=1}^{P}g(f(s^k_{i&plus;pAN}))" title="\huge e^k_{i+AN} = \frac{1}{P}\sum_{p=1}^{P}g(f(s^k_{i+pAN}))" /></a>
</p align="center">


<!--    
<p align="center">
  <img src="https://github.com/ghafeleb/goodfellas/blob/main/docs/resources/AvgPosSpan.PNG" width="450" /> 
</p>
 -->
Now we have <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;2(AN)" target="_blank"><img src="https://latex.codecogs.com/png.latex?\small&space;2(AN)" title="\small 2(AN)" /></a> datapoints per party and in total <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;4(AN)" target="_blank"><img src="https://latex.codecogs.com/png.latex?\small&space;4(AN)" title="\small 4(AN)" /></a> datapoints per batch. 

<p align="center">
  <img src="https://github.com/ghafeleb/goodfellas/blob/main/docs/resources/Loss1.PNG" width="450" /> 
</p>
<br>
<p align="center">
  <img src="https://github.com/ghafeleb/goodfellas/blob/main/docs/resources/Loss2.PNG" width="450" /> 
</p>

where

<p align="center">
  <img src="https://github.com/ghafeleb/goodfellas/blob/main/docs/resources/Loss3.PNG" width="450" /> 
</p>


\ali{What is "sim" in the formulation?, what is $\tau$?} Loss function~\eqref{eq:loss_k} enforces anchor $e^k_i$ to be as closes as possible to its corresponding positive span $e^k_{i+AN}$ (which is referred to as easy positive) and at the same time to be as far as possible from all spans $e^{1-k}_m$ from the opposite party, i.e., for any given anchor from class $k$, the corresponding set of negative spans only include the spans from opposite class $1-k$ (which are referred to as easy negative). Figure~\ref{fig:model} visualizes a simplified overview of our model. \ali{Are we using hard positive or hard negative in our project? If no, I think we should remove the term "easy".}
\zahra{What is m? Does it refer to all documents from the different parties?}

<p align="center">
  <a href='https://www.linkpicture.com/view.php?img=LPic5fc59238d28c21286633194'><img src='https://www.linkpicture.com/q/embedding.png' type='image'></a>
</p>
<p align="center">
<b>Figure 2:</b> Overview of the supervised contrastive objective. In this figure, we show a simplified example where in each batch we sample 1 document $d^k$ per class $k$ and we sample 1 anchor span $e^k_i$ per document and 1 positive span $e^k_j$ per anchor. All the spans are fed through the same encoder $f$ and pooler $g$ to produce the corresponding embedding vectors $e^k_i$ and $e^k_j$. The model is trained to minimize the distance between each anchor $e^k_i$ and its corresponding positive $e^k_j$ and maximize the distance between anchor $e^k_i$ and all other spans from class $1-k$. \ali{I think we shouk}
</p>
<p align="center">
  <a href='https://www.linkpicture.com/view.php?img=LPic5fc59316dc5c02069018779'><img src='https://www.linkpicture.com/q/data_head.png' type='image'></a>
</p align="center">
<p  align="center">
<b>Figure 3:</b> Overview of the dataset.
</p>


<p align="center">
  <a href='https://www.linkpicture.com/view.php?img=LPic5fc59316dc5c02069018779'><img src='https://www.linkpicture.com/q/data_head.png' type='image'></a>
</p>
<p  align="center">
<b>Figure 3:</b> Overview of the dataset.
</p>


## Data
For our experiments we use [AYLIEN’s Coronavirus news dataset](https://aylien.com/blog/coronavirus-news-dashboard) (Global COVID related news since Jan 2020). This dataset contains numerous news articles from different news sources with different political orientations. For simplicity we only focus on news articles from two news sources Huffington Post, which is considered as liberal (class $0$), and Breitbart which is considered as conservative (class $1$).

In the figure~\ref{fig:data_head} we show the first few lines of the dataset. We assign Huffington's articles class $0$ and Breitbart's articles class $1$. Another important observation from the data is the distribution of the length (number of words) of the articles which is shown in figure~\ref{fig:data_head}. This is important to the step where we sample the anchor-positive pairs from the data. 

<p align="center">
  <img src="https://github.com/ghafeleb/goodfellas/blob/main/docs/resources/data_head.png" width="900" /> 
</p align="center">
<p  align="center">
<b>Figure 3:</b> Overview of the dataset.
</p>

<p align="center">
  <img src="https://github.com/ghafeleb/goodfellas/blob/main/docs/resources/length_distribution.png" width="450" /> 
</p>
<p  align="center">
<b>Figure 3:</b> Overview of the dataset.
</p>

Another step that we do is topic modeling to make sure all the articles are about the same subject ''covid19''. We use Latent Dirichlet Allocation (LDA) for this step. The topics we found are as follows:

- Huffpost people new time home like 19 covid pandemic health help year just
- Trump president donald people house states white pandemic news state virus americans health going huffpost
- Minister china chinese cases italy wuhan government confirmed border reported countries virus prime authorities deaths
- Hanks rita kimmel jimmy wilson cordero aniston kloots fallon elvis song tom actor conan corden
- Newstex al views content et https advice www accuracy commentary authoritative guarantees distributors huffington conferring

It seems that the first topic is about ''covid19'', the second topic is about ''white house announcements'', the third one is about ''global news'', the fourth one is about ''enterntainment'' and the last one is not related to our work. To only keep covid19 related articles we kepth those having at least one of the following keywords, ''covid,covid19,pandemic,vaccine,virus,corona,face covering''.
At the end we are left with 7226 articles from Breitbart (class $1$) and 6300 articles from Huffington Post (class $0$).

## Experiments
In this section, as one of our baseline methods we train the \texttt{DeCLUTR} model introduced by[[3]](#3) on our covid19 data explained in \S\ref{sec:data}, the overview of their model is given in figure 3. 

<p align="center">
  <img src="https://github.com/ghafeleb/goodfellas/blob/main/docs/resources/DeCUTR.PNG" width="900" /> 
</p align="center">
<p  align="center">
<b>Figure 4:</b> Overview of the self-supervised contrastive objective. For each document d in a minibatch of size N, we sample A anchor spans per document and P positive spans per anchor. For simplicity, we illustrate the case where $A = P = 1$ and denote the anchor-positive span pair as $s_i$, $s_j$. Both spans are fed through the same encoder $f$ and pooler $g$ to produce the corresponding embeddings $e_i = g(f(s_i))$, $e_j = g(f(s_j))$. The encoder and pooler are trained to minimize the distance between embeddings via a contrastive prediction task (where the other embeddings in a minibatch are treated as negatives, omitted here for simplicity).
</p>

In the implementation of \texttt{DeCLUTR}, in the process of sampling the anchor-positive spans, they randomly choose the length of each span, with the minimum length $l_{\min}=32$ and maximum length $l_{\max}=512$. Furthermore they exclude all articles with less than ($\text{num-anchor}*l_{\max}*2$) words, where num-anchor is the number of anchors sampled per article (For details of the sampling process please refer to the main text of~\cite{giorgi2020declutr}). According to the distriubtion of length of the articles in our dataset given in figure~\ref{fig:length_dist}, in order to be able to use most of our data, we set the minimum length to $l_{\min}=20$ and maximum length to $l_{\max}=100$ and we sample one anchor per document, i.e., $\text{num-anchor}=1$. Having all this, articles with less than $200$ words (1345 of them) would be put aside which we use them as our test set and use the remaining articles (12181 of them) as our training set.
We train the \texttt{DeCLUTR} model with the unsupervised contrastive loss on the training data. We then get the embedding of the test articles under the trained model. The visualization of the embeddings is given in figure~\ref{fig:pca} (left). The embedding space is $768$ dimensional. We applied Principal component analysis (PCA) to get the visualization. As we can see the embeddings are not well-separated from each other.
As our next step, we fit a binary classification model on these embeddings to see how well it can separate the articles from opposite classes. To do so, we fit a logistic regression model on $75\%$ of the test set. The accuracy of the trained binary classifier on the remaining $25\%$ of the data is $85.45\%$.  

<p float="center">
  <img src="https://github.com/ghafeleb/goodfellas/blob/main/docs/resources/declutr_pca.jpg" width="450" /> 
  <img src="https://github.com/ghafeleb/goodfellas/blob/main/docs/resources/fineBERT_pca.png" width="450" />
</p>
<p  align="center">
<b>Figure 4:</b> The visualization of the embeddings from \texttt{DeCLUTR} (left) and \texttt{FineBERT} (right) of test data in two dimension.
</p>


As our second baseline method, we fine tune BERT~\cite{mikolov2013distributed} by adding a classification layer and minimizing the the classification loss. We refer to this approach as \texttt{FineBERT}. A visualization of  the model \texttt{FineBERT} is given in figure~\ref{fig:FineBERT}.

Similar to \texttt{DeCLUTR}, we visualize the embeddings given by \texttt{FineBERT} in two dimension shown in figure~\ref{fig:pca} (right). Similar to the case \texttt{DeCLUTR} we fit a logistic regression model on $75\%$ of the test set. The accuracy of the trained binary classifier on the remaining $25\%$ of the data is $55.78\%$. \texttt{FineBERT} is performing worse than \texttt{DeCLUTR} which shows the power of the self-supervised contrastive loss.

So far we have implemented the baseline methods \texttt{DeCLUTR} and \texttt{FineBERT} and we can see that there is room for improvement. The out of sample accuracy of the downstream classification task is not good and we believe that our proposed model \texttt{GoodFellas} would improve upon that.

<p align="center">
  <a href='https://www.linkpicture.com/view.php?img=LPic5fc5916db37a31261281140'><img src='https://www.linkpicture.com/q/FineTunedBERT.png' type='image'></a>
</p>
<p  align="center">
<b>Figure 7</b>
</p>

## References
<a id="1">[1]</a> 
Lucas Dixon, John Li, Jeffrey Sorensen, Nithum Thain, and Lucy Vasserman. Measuring and mitigating unintended bias in text classification. In proceedings of the 2018AAAI/ACM Conference on AI, Ethics, and Society, pages 67–73, 2018.

<a id="3">[3]</a> 
John M Giorgi, Osvald Nitski, Gary D Bader, and Bo Wang. Declutr: Deep contrastivelearning for unsupervised textual representations.arXiv preprint arXiv:2006.03659, 2020.

<a id="4">[4]</a> 
Jon Green, Jared Edgerton, Daniel Naftel, Kelsey Shoub, and Skyler J. Cranmer. Elusiveconsensus:  Polarization in elite communication on the COVID-19 pandemic. Science Advances, 6(28):eabc2717, July 2020.

<a id="6">[6]</a> 
Negar Mokhberian, Andrés Abeliuk, Patrick Cummings, and Kristina Lerman. Moralframing and ideological bias of news.arXiv preprint arXiv:2009.12979, 2020.

<a id="5">[5]</a> 
Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. Distributed representations of words and phrases and their compositionality. In advances in neural information processing systems, pages 3111–3119, 2013.






<!-- 
## Bias detection using Deep Supervised Contrastive Learning (Goodfellas)

You can use the [editor on GitHub](https://github.com/ghafeleb/goodfellas.github.io/edit/gh-pages/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/ghafeleb/goodfellas.github.io/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.

 -->
