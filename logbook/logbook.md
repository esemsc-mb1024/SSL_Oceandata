# Logbook
Supervisor Meeting Notes 1
Date: 13 June 2025 
Attendees:
- Bjorn Rommen
- Matthew Barrett
Discussion Topics
1. Data Engineering
- Identified which Sentinel-1 data products to download
- Defined preprocessing steps needed to extract relevant and up-to-date data
- Discussed potential use of additional data sources to enrich the dataset

2. Geophysical Process Analysis
- Explored the influence of geophysical processes in SAR data
- Considered methods for handling these effects during model training
3. Initial Model Design
- Reviewed initial ideas for neural network architecture
- Discussed suitability of proposed designs for handling SAR data characteristics

Feedback
- Project is progressing well
- Begin outlining a timeline for major project milestones
- Clarify the expected deliverables, including evaluation metrics and results

Plan for Next Meeting
- Build and test an initial version of the working model
- Finalise the dataset downloading and preprocessing pipeline


Supervisor Meeting Notes 2
Date 28th June 
Attendees: 
- Bjorn Rommen
- Andreas Theodosiou
- Matthew Barrett

Discussion Topics
1. Data Engineering
- Clear pipeline Created to establish a way to download and use 50,000 sar wavemode images, an appropriate amount of images to use for this task

2. Model Design
- A SSL model has was proposed and discussed a the model that will be used to try and classify these features

Plan for next meeting 
- Deliver some results of the clustering which is the intended result of the research


Supervisor meeting 3
Date 09th July
Attendees:
- Bjorn Rommen
- Andreas Theodosiou
- Owen O Driscoll
- Matthew Barrett

Discussion Topics
1. Review of Current Results
- Observing clusters as to see what the model is currently identifying to assign to each cluster.
- Evaluation of how well they are clustering

2. Normalisation
- Discussing wether the current method of local normalisation is viable for the goal of the project

Plan for next meeting 
- Eplore other avenues of evaluation


Supervisor meeting 4
Date 16th of July 
Attendees:
- Bjorn Rommen
- Matthew Barrett

Discussion Topics
1. New method of Results
- Introduced a small labelled dataset to carry out logistic regression using my model
- Discussed the value of results in relation to the manually annotated dataset

2. Explainable AI
- Ensure that you evalute each step of the process and can explain what your model is producing the results it is producing for each geophysical feature

Plan for Next Meeting 
- Use the new method of results as a score metric to improve the model
- Use the full dataset to train model and finalise presentable results
  


