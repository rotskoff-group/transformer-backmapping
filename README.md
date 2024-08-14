# Data-Efficient Generation of Protein Conformational Ensembles with Backbone-to-Side-Chain Transformers

Official implementation of:  

**Data-Efficient Generation of Protein Conformational Ensembles with Backbone-to-Side-Chain Transformers**

Shriram Chennakesavalu and Grant M. Rotskoff

<https://pubs.acs.org/doi/full/10.1021/acs.jpcb.3c08195>

**Abstract**: Excitement at the prospect of using data-driven generative models to sample configurational ensembles of biomolecular systems stems from the extraordinary success of these models on a diverse set of high-dimensional sampling tasks. Unlike image generation or even the closely related problem of protein structure prediction, there are not currently data sources with sufficient breadth to parameterize generative models for conformational ensembles. To enable discovery, a fundamentally different approach to building generative models is required: models should be able to propose rare, albeit physical, conformations that may not arise in even the largest data sets. Here we introduce a modular strategy to generate conformations based on "backmapping" from a fixed protein backbone that 1) maintains conformational diversity of the side chains and 2) couples the side chain fluctuations using global information about the protein conformation. Our model combines simple statistical models of side chain conformations based on rotamer libraries with the now ubiquitous transformer architecture to sample with atomistic accuracy. Together, these ingredients provide a strategy for rapid data acquistion and hence a crucial ingredient for scalable physical simulation with generative neural networks.
