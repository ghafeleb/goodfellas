## Bias detection using Deep Supervised Contrastive Learning (Goodfellas)
#### [[Project Website]](https://ghafeleb.github.io/goodfellas/)


[Sina Aghaei](saghaei@usc.edu)<sup>1</sup>, [Zahra Abrishami](zabrisha@usc.edu)<sup>1</sup>, 
[Ali Ghafelehbash](ghafeleb@usc.edu)<sup>1</sup>, [Bahareh Harandizadeh](harandiz@usc.edu)<sup>1</sup>, [Negar Mokhberian](nmokhber@usc.edu)<sup>1</sup>


<sup>1</sup>CLVR Lab, University of Southern California 


## Abstract
In this paper, we propose an end-to-end model to detect ideological bias in news articles. We propose a deep supervised contrastive learning model to learn new representations for text with the goal of separating the embeddings from different classes to expose the bias in the textual data and exploit this bias to identify political orientation of different news articles.

## Introduction

In any nation, media can be a dividing issue as it might reflect topics differently based on the political views or ideological lines. Understanding this implicit bias is becoming more and more critical, specifically by growing numbers of social media and news agencies platforms. Recently the number of research in this domain also is increasing, from detection and mitigation of gender bias~\cite{dixon2018measuring},to polarization detection in political views~\cite{2020SciA6C2717G} also ideological bias of News~\cite{mokhberian2020moral}. We think the analysis of bias in the news could be very helpful for the readers as make them more responsible about what they hear or read. 

In this work we want to understand how different are news articles on the same subject but from different political parties (left and right) from each other. We want to detect the potential political bias within news articles. We, as human, can easily identify the different political orientation of articles from opposite parties. For example, how different conservative news agencies such as ''Fox News'' approach a subject like Covid-19 compares to a liberal news agency such as ''CNN''. The question is that can machines detect this political bias as well?
%
A proxy for this goal could be a classifier which tries to classify news articles depending on their political party. Existing approaches such as~\cite{mokhberian2020moral} tackle this problem using a classifier on the space of the words embedding. The problem with this approach is that it is not end to end, i.e., the embedding are not trained with the purpose of getting a good classification result. As we can see in figure~\ref{fig:bias-embedding}(right), with general purpose word embedding models such as BERT~\cite{mikolov2013distributed}, classifying embedded articles might not be straightforward. Having a new representation such as the one shown in figure~\ref{fig:bias-embedding}(left) where it maximizes the distance between embedding from different classes could make the classification task much easier, as in the latent space, the bias is exposed.

<p align="left">
  <img  width="250" src="https://github.com/ghafeleb/goodfellas/blob/main/docs/resources/bias.PNG"> 
<!--   <img align="center" width="250" src="https://github.com/ghafeleb/goodfellas/blob/main/docs/resources/bias.PNG">  -->
  <img align="right" width="250" src="https://github.com/ghafeleb/goodfellas/blob/main/docs/resources/bias.PNG">
</p>
<!-- <img align="left" width="100" src="https://github.com/ghafeleb/goodfellas/blob/main/docs/resources/bias.PNG"> -->









To achieve such representation for news articles we propose a modification to the deep contrastive Learning model for unsupervised textual representation introduced in~\cite{giorgi2020declutr}. In~\cite{giorgi2020declutr}, they have a unsupervised contrastive loss which for any given textual segment (aka anchor span) it minimizes the distance between its embedding and the embeddings of other textual segments randomly sampled from nearby in the same document (aka positive spans). It also maximizes the distance between the given anchor from other spans which are not in its neighborhood (aka negative spans). In their model, the positive and negative spans are not chosen according to the label of the documents. We propose to alter their objective to a supervised contrastive loss so that the negative spans are sampled from articles with opposite label. The motivation is to maximize the distance between articles from different classes.



## Problem Formulation














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
