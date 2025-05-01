# HDM_Python

# Setup

- Install requirements with `pip install -r requirements.txt`
- If running on gpu, then install pytorch

# Running HDM on platyrrhine teeth

In order to run, add the data prepared by Rob. The data directory should be inside HDM_Python, and needs to be named "platyrrhine".
The platyrrhine folder should contain:

- Names.mat
- FinalDist.mat
- softMapMatrix.mat
- ReparametrizedOFF with the .off files inside.

By running `python src/HDM_teeth_CPU.py` from within the src folder, you should get the pringle below.

Proof of Concept Pringle (PCP):

![](pringle.png)

Times:

- HDM_CPU.py: 101.45s user 18.68s system 505% cpu 23.776 total
