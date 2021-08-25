# waterleidingen
Project from Esri Nederland --- Prediction of water pipe leaks in Limburg by developing a solution based primarily on machine-learning tools in ArcGIS 
---
This project is an internship which is one of the projects of the GeoAI department in Esri Nederland. It is placed with the one client’s intention of applying GeoAI techniques to optimize the replacement of pipes prone to failures.  

A general assumption in this research is that all pipes will fail eventually. With this in mind, the objective of this internship is to develop a solution to predict the pipes with a particular problem − pipe leak (the major problem in the study area) and when the problem occurs. The solution is primarily based on the toolboxes available in ArcGIS with the aid of open-source programming languages. In this way, this internship also explores the possibility to use the machine learning tools in ArcGIS and machine learning functions in open-source languages in tandem. Besides, this topic concerns drinking water companies’ current struggle  ̶  providing sustainable and standard drinking water service which is essential for users (e.g., domestics users) to access safe drinking water.

The measured variables (i.e., environmental variables, intrainsic variables and operational variables) in this internsip are shown in Table 1 and 2. 

Table 1: Intrinsic and operational environmental variable measured in this internship. 
|Categories    |  Variables         |  
| ------------- |:-------------:| 
|Intrinsic variables (at the level of per pipe)        | Life expectancy           | 
|      | Material*           | 
|      | Diameter           | 
|      | Number of aftakking (without the type zadel)          | 
|      | Number of bocht           | 
|      | Number of verloop           | 
|Operational variables (at the level of per pipe)      | Old afsluiter (if the afsluiter is older than the average age)*          | 
|      | Number of overgang          | 

*: Different kinds of materials (e.g., whether the pipe is made of AC or CU, Fig. 1) and old afsluisters were put into the model as binomial variables.

<img width="178" alt="Capture3" src="https://user-images.githubusercontent.com/87322555/125421313-51b6514f-86b9-41af-ac31-5e909dddf1b4.PNG">

Figure 1: Pipes made of different kind of materials

Table 2: Environmental variables measured in this intership.
| Categories        | Environmental variables (at the level of per pipe)          | 
| ------------- |:-------------:|  
| Trees      | Numbers of trees |
|      |Presence of deciduous trees*         |  
|  | Maximum values of root system radius of all surrounding trees| 
| Roads (funtion)     | Roads allowed to drive automobiles*         | 
|      |    Roads allowed to cycle*       | 
|      |  Pavement*         | 
|  | Parking area*| 
|      |     Local*     | 
|      |    Regional*        | 
|      |   Others (e.g., roads allowed to ride horses)*         | 
| Roads (physical status) | Closed*| 
|      |     Semi-closed*     | 
|      |    Paved*        | 
|      |   Unpaved*         |  
| Soil type| Clay*| 
|      |     Sand and clay*     | 
|      |    Human influence*        | 
|      |   Sand*         | 
|      | Loam*| 
|      |     Peat*     | 
|      |    Limestone*        | 
|      |   Water*         | 
|      |   Dike and ining fleid*         | 
| Slope     | Slope of water pipes  | 
|      |    Whether the pipe sits in hilly area*     | 
|Sink area     |    water pipes located in areas where water likely to store        | 
|  Address  |   Number of addresses         | 

*: Binominal variables.  

The scripts written when developing the solution were included in the files in this project, in which two scripts contains Baysian optimization function. Hereby the ciation of the Baysian optimziation:
@Misc{,
    author = {Fernando Nogueira},
    title = {{Bayesian Optimization}: Open source constrained global optimization tool for {Python}},
    year = {2014--},
    url = " https://github.com/fmfn/BayesianOptimization"
}

