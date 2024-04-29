# ML-GWBSE
Python codes to develop ML models using our GW-BSE database (https://hydrogen.cmd.lab.asu.edu/) and applying it with Materials Project (https://next-gen.materialsproject.org/) data. 

Description of files:

1. funcs.py: All the internal functions are defined here.
2. prep_mydata.py: Code to retrieve data from computed GW-BSE database
3. prep_mpdata.py: Code to retrieve data from the Materials Project database using MP API-key
4. ml_regression.py: Code to develop and use the regression ML models
5. ml_classification.py: Code to develop and use the classification ML models
6. check_conv.py: Example of using ML regression routines to check convergence with several parameters
7. classify.py: Example of using ML classification routines to use different classification cutoff parameters
8. comp_anlz.py: Code to analyze the starting dataset to classify them based on spacegroup etc.

## Citation
If you use these codes, please cite
1. T. Biswas, and A. K. Singh. "Incorporating quasiparticle and excitonic properties into material discovery." arXiv preprint arXiv:2401.17831 (2024).
2. T. Biswas, and A. K. Singh. "py GWBSE: a high throughput workflow package for GW-BSE calculations." npj Computational Materials 9.1, 22 (2023).
