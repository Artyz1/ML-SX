ML-SX: machine learning for the modelling of metal solvent extraction processes 

Solvent extraction is a crucial process to purify metals from either primary or secondary sources. 
Modelling solvent extraction processes is a cumbersome task requiring extensive experimental work to understand the behaviour of the metal in the aqueous and organic phases. 
A database containing more than 18,000 data points related to the extraction of 72 metals and non-metals and 5 acids extracted with tributyl phosphate (TBP) has been built and is used to train ML models.


The most accurate ML model (Extra Trees Regressor) allows modelling the extraction of acids and metals into organic phases involving TBP, as long as sufficient and reliable experimental data are available. 
However, it fails to model accurately some solvent extraction systems due to the lack of experimental data points and conflicts between various experimental values obtained from different references. 
In that case, a neural network based on the Levenberg-Marquardt algorithm is more appropriate to deal with small datasets.  
The Extra Trees Regressor model developed has a good generalisation ability, it is able in most cases to predict extraction efficiency of metals or acids not included in the training or testing datasets. 

