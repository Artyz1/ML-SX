ML-SX: machine learning for the modelling of metal solvent extraction processes 

Solvent extraction is a crucial process to purify metals from either primary or secondary sources. Modelling solvent extraction processes is a cumbersome task requiring extensive experimental work to understand the behaviour of the metal in the aqueous and organic phases. 
A database containing around than 18,800 data points related to the extraction of 72 metals and non-metals and 5 acids extracted with tributyl phosphate (TBP) has been built and is used to train machine learning (ML) models. The database can be found in the "Database solvent extraction with TBP". The literature references to build the database can be found in the excel file. 

Data on the extraction of metals and acids were gathered by selecting studies where TBP is used (alone) as an extracting molecule. Since TBP has been used at the industrial scale for decades, data for multiple metals in multiple aqueous media are available (metals, nonmetals and acids extracted from sulfuric, phosphoric, hydrochloric, hydrobromic, perchloric and fluoric acid media).  The periodic table displayed in Fig. 1 shows the metals, non-metals and protons (H) for which data is available, and the anion used in the aqueous solution considered.

![image](https://github.com/user-attachments/assets/c27a9284-414e-4757-b724-42c5117f33e5)

Figure 1. Periodic table of the elements included in the dataset. Different colours indicate the nature of the anion in the aqueous phase

The dataset regroups data for different oxidation states of the same metal when available, e.g. Cr(VI) and Cr(III), Cu(I) and Cu(II), or Pu(III) and Pu(VI). In most studies, TBP is used undiluted or diluted in conventional hydrocarbon diluents such as kerosene, heptane or dodecane, but the dataset also contains data for solvents such as carbon tetrachloride or chloroform. The features were selected in order to describe the extraction system as precisely as possible. Nature of the element extracted is represented by its oxidation state and its crystal radius. The crystal radius was selected based on the oxidation state of the element extracted. When multiple values are available, the value selected is based on the (arbitrarily chosen) coordination number of +VI (and high spin for Co(II), Fe(II) and Mn(II)). Other features relative to the aqueous phase are the acid concentration (in mol/L), the initial metal concentration (in g/L), and the features relative to the anion (crystal radius for +VI coordination number, and oxidation state). In addition, the anion’s concentration is corrected if a salt is present in the media. Different salts having different effects on the activity coefficients, the nature of the cation of the salt is taken into account (using its crystal radius). If another anion is present in the aqueous phase and has a significant effect on the extractability of the target element (e.g. addition of HF suppresses the extraction of uranium and protactinium from chloride media), its concentration and crystal radius are added. 

Regarding the organic phase, the concentration of TBP (%) is used as a feature, as well as the dielectric constant (ε) of the solvent used, since it defines the intensity and nature of interaction involved during the solvent extraction of metals. Dielectric constant of TBP (ε=8.34) is used when undiluted TBP is used as an organic phase. Other features include the O/A ratio (equal to 1 in most cases) and the mixing time (seconds). The kinetics aspects of the solvent extraction are not studied here since the all of the data included in the database are reported at equilibrium, but further work will be dedicated to the study of those kinetics aspects, using more relevant features such as viscosity of the phases and hydrodynamic conditions. Temperature was not selected as a feature since most of the experimentally determined values have been obtained at room temperature (25°C). 

8 different machine learning algorithms were tested on this dataset, their hyperparameters were optimised using Gridsearch. The target value is the extraction efficiencies ("E"). By using this value instead of the distribution ratio, the ML algorithms were found to perform better to a narrower distribution of the target values. Further tuning of the ML algorithms was carried out with E values between 0.1 and 0.99, which further limits the range and eliminates negligible extraction (below 1%) or complete extraction (higher than 99%) of the considered species. 

The best performing algorithm was found to be Extrtrees regressor, which shows the highest R squared but also the lowest mean absolute error (AME), root mean square error (RMSE), and average absolute relative deviation (see Table 1). All algorithms show a slight tendency to overfitting with poorer metrics during testing as compared to training, in particular the extratrees regressor with a significantly higher testing AARD. 

Table 1. Comparison of the metrics obtained with different optimised ML algorithms 
![image](https://github.com/user-attachments/assets/efcc189a-ae9e-418e-bc1c-8514a4af5604)

The performance of the ExtraTreesRegressor was significantly improved by removing outliers using an Isolation Forest (see Table 2 and Figure 2), allowing to obatin a training AARD as low as 2.6%.

Table 2. Optimised hyperârameters and performances obtained with the extratrees regressor
![image](https://github.com/user-attachments/assets/5fc89864-fdc5-4b83-b487-56d741a5c897)



![image](https://github.com/user-attachments/assets/67298472-8602-4630-b5e5-5b10472ed952)
Figure 2. Experimental vs calculated E using the opitmised extratrees regressor


This version could then be used to predict extraction efficiency values. Predictions are obviosuly dependent on the training data and could be innacurate, in particular if conflicts exists between literature data. Figure 3 shows an example of the predicted vs actual extraction of acids into an organic phase made of- undiluted TBP.




![image](https://github.com/user-attachments/assets/0e010383-107d-4434-9bce-ea9453ac335c)
![image](https://github.com/user-attachments/assets/7fb5b452-88dc-45ff-b950-7fb7163b7d34)
![image](https://github.com/user-attachments/assets/fd0b886c-42ed-46d3-8982-a2814e4afe81)
![image](https://github.com/user-attachments/assets/d76e02d6-69df-495d-bb7a-e536a5b857a3)
Figure 3. Predicted (line) and experimental (dots) concentration of various acids extracted in the pure TBP phase as a function of the initial aqueous acid concentration. 
