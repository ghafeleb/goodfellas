## Bias detection using Deep Supervised Contrastive Learning (Goodfellas)
#### [[Project Website]](https://ghafeleb.github.io/goodfellas/)


[Sina Aghaei](mailto:saghaei@usc.edu)<sup>1</sup>, [Zahra Abrishami](mailto:zabrisha@usc.edu)<sup>2</sup>, 
[Ali Ghafelehbash](mailto:ghafeleb@usc.edu)<sup>1</sup>, [Bahareh Harandizadeh](mailto:harandiz@usc.edu)<sup>2</sup>, [Negar Mokhberian](mailto:nmokhber@usc.edu)<sup>2</sup>

<sup>1</sup>Department of Industrial and Systems Engineering, University of Southern California, Los Angeles, CA 9008<br/>
<sup>2</sup>Department of Computer Science, University of Southern California, Los Angeles, CA 9008


## Abstract
In this paper, we propose an end-to-end model to detect ideological bias in news articles. We propose a deep supervised contrastive learning model to learn new representations for text with the goal of separating the embeddings from different classes to expose the bias in the textual data and exploit this bias to identify political orientation of different news articles.

## Introduction

In any nation, media can be a dividing issue as it might reflect topics differently based on the political views or ideological lines. Understanding this implicit bias is becoming more and more critical, specifically by growing numbers of social media and news agencies platforms. Recently the number of research in this domain also is increasing, from detection and mitigation of gender bias[[1]](#1), to polarization detection in political views[[4]](#4) also ideological bias of News[[6]](#6). We think the analysis of bias in the news could be very helpful for the readers as make them more responsible about what they hear or read. 

In this work we want to understand how different are news articles on the same subject but from different political parties (left and right) from each other. We want to detect the potential political bias within news articles. We, as human, can easily identify the different political orientation of articles from opposite parties. For example, how different conservative news agencies such as ''Fox News'' approach a subject like Covid-19 compares to a liberal news agency such as ''CNN''. The question is that can machines detect this political bias as well?
%
A proxy for this goal could be a classifier which tries to classify news articles depending on their political party. Existing approaches such as[[6]](#6) tackle this problem using a classifier on the space of the words embedding. The problem with this approach is that it is not end to end, i.e., the embedding are not trained with the purpose of getting a good classification result. As we can see in figure~\ref{fig:bias-embedding}(right), with general purpose word embedding models such as BERT[[5]](#5), classifying embedded articles might not be straightforward. Having a new representation such as the one shown in figure~\ref{fig:bias-embedding}(left) where it maximizes the distance between embedding from different classes could make the classification task much easier, as in the latent space, the bias is exposed.


<p float="center">
  
  <img src="https://github.com/ghafeleb/goodfellas/blob/main/docs/resources/embedding.PNG" width="450" /> 
  <img src="https://github.com/ghafeleb/goodfellas/blob/main/docs/resources/bias.PNG" width="450" />
  <!--   <img src="/docs/resources/embedding.PNG" width="300" /> -->
<!--   <figcaption>{{An ideal latent space (left) where the articles from opposite classes are far from each other which helps to expose the political bias (right) and improves the performance of the classification task.}}</figcaption> -->
</p>
*Fig. 2: The minimum dominating set of a graph*







To achieve such representation for news articles we propose a modification to the deep contrastive Learning model for unsupervised textual representation introduced in~\cite{giorgi2020declutr}. In~\cite{giorgi2020declutr}, they have a unsupervised contrastive loss which for any given textual segment (aka anchor span) it minimizes the distance between its embedding and the embeddings of other textual segments randomly sampled from nearby in the same document (aka positive spans). It also maximizes the distance between the given anchor from other spans which are not in its neighborhood (aka negative spans). In their model, the positive and negative spans are not chosen according to the label of the documents. We propose to alter their objective to a supervised contrastive loss so that the negative spans are sampled from articles with opposite label. The motivation is to maximize the distance between articles from different classes.



## Problem Formulation
We consider a setting where we have various documents (articles) from two different parties called \emph{liberal} (label being 0) and \emph{conservative} (label being 1). All the documents are about a similar topic, \newsa{Covid-19}.

We sample a batch of $N$ documents from the \emph{liberal} party (class label being 0) and $N$ documents from the \emph{conservative} party (class label being 1). For each document from class $k \in \{0,1\}$ we sample $A$ anchor spans $s^k_i,~ i \in \{1,\dots,AN\}$ and per anchor we sample $P$ positive spans $s^k_{i+pAN},~ p \in \{1,\dots,P\}$ following the procedure introduced in~\cite{giorgi2020declutr}. \ali{Isn't it better to have $s_{i, j}^k$ such that $i \in \{1, 2, ..., N\}$ and $j \in \{1, 2, ..., A\}$?}

Given an input span, $s^k_i$, a ''transformer-based language models'' encoder $f$, maps each token in the input span $s^k_i$ to a word embedding.
    
Similar to~\cite{giorgi2020declutr}, a pooler $g(.)$, maps the encoded anchor spans $f(s^k_i)$ to a fixed length embedding $g(f(s^k_i))$. \ali{Is it possible to impose fixed length of output to our embedder?}
    
We take the average of the positive spans per anchor as follows
    $$e^k_{i+AN} = \frac{1}{P}\sum_{p=1}^{P}g(f(s^k_{i+pAN}))$$
    




Ali Ghafelebashi[[1]](#1).
Zahra Abrishami[[2]](#2).

## References
<a id="1">[1]</a> 
Lucas Dixon, John Li, Jeffrey Sorensen, Nithum Thain, and Lucy Vasserman. Measuring and mitigating unintended bias in text classification. In proceedings of the 2018AAAI/ACM Conference on AI, Ethics, and Society, pages 67–73, 2018

<a id="4">[4]</a> 
Jon Green, Jared Edgerton, Daniel Naftel, Kelsey Shoub, and Skyler J. Cranmer. Elusiveconsensus:  Polarization in elite communication on the COVID-19 pandemic. Science Advances, 6(28):eabc2717, July 2020

<a id="6">[6]</a> 
Negar Mokhberian, Andrés Abeliuk, Patrick Cummings, and Kristina Lerman. Moralframing and ideological bias of news.arXiv preprint arXiv:2009.12979, 2020.

<a id="5">[5]</a> 
Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. Distributed representations of words and phrases and their compositionality. In advances in neural information processing systems, pages 3111–3119, 2013


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
