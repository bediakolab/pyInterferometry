
# pyInterferometry - Python Bragg Interferometry Code
Python code for preforming Bragg Interferometry, a data analysis technique for 4D-STEM of moiré materials as described in Refs [1,2].
Implemented for and used in Ref [2].

## Citation
Please see and cite the following manuscripts for a description of the method.

1: Nathanael P. Kazmierczak, Madeline Van Winkle, et al. Nature materials 20.7 (2021): 956-963.

2: Madeline Van Winkle, Isaac M. Craig, et al. Nat Commun 14, 2989 (2023). (In particular the SI of this work described many details of this codebase)

## Contact
Please contact me with any questions, issues, or requests.
Isaac M. Craig - craigi[at]berkeley[dot]edu

## Quick Usage
format your 4d-stem data (gatan dm4 and/or py4dstem formatted h5 accepted) in accordance with the example input directory. 
Then place a Diffraction_SI.dm4 or dp.h5 in each folder. 
You can then run the code from the driver script "main.py" which will guide you through the analysis workflow, asking for user input when needed.
All processed data and images will be saved in the corresponding data/Material_Name/dsN directories (where N is the dataset number). 

## Examples
An example of a processed dataset is given in data/MoS2_AP/ds6. 
A raw Diffraction_SI.dm4 and processed diskset.pkl files used in Ref [2] are available on https://zenodo.org/record/7865970
