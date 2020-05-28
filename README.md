# SLD Reassessment
This repository holds the Python code used to run the experiments
presented and discussed in _Andrea Esuli, Alessio Molinari, and Fabrizio Sebastiani. 2020. A Critical Reassessment of the Saerens-Latinne-Decaestecker Algorithm for Posterior Probability Adjustment.ACM Transactions on Information Systems1, 1(May 2020), 32 pages_.

## How to run the experiments
### Prerequisites
In order to run the experiments you will need a working installation of the latest 
stable Python release (currently 3.8). It is recommended to have the `pip` package manager
installed as well. 
  
You can install all the required packages by cloning this repository and run:  
`pip install -r requirements.txt`  

### Running
Run the experiments by executing the `sld_experiments.py` file with your Python 
interpreter. By default, the code will run binary classification experiments. You 
might want to change this and a few other details before actually running the experiments:  
- In the `main` block, at the end of the file, you should either call the `binary_experiments()`
or the `single_label_experiments()` functions;
- In the above mentioned functions, you might want to change the number of processes to create when 
running (by default, it runs with 11 processes but this might not be suited for your 
machine). Simply change the number of processes in the `Pool` object in both 
functions.
- Finally, you might want to change how many experiments are run for each classifier. You 
can adjust this in the `ITERATIONS_NUMBER` variable at the begin of the file.  
  
If everything looks fine, you can finally run the experiments. Notice that the program is logging 
output on a `computation.log` file and saving computation results in the `pickles/measures_new_experiments`
directory. As a consequence, make sure the user running the program has read/write permission in the working directory.


## Generating plots as seen in the visualization tool
In order to generate plots as you might have seen in the [visualization tool](https://hlt-isti.github.io/SLD-visualization/)
you can run the `export_html_for_visualizer.py` file. This program accepts several command line options: 
simply run the program with  
`python export_html_for_visualizer.py -h`  
to read how to properly generate the plots you wish to export.