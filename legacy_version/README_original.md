# Extended Phase-Field KKS model: A Phase-field model for growth and coarsening of Si precipitate in AlSi10Mg SLM/LPBF in a super-saturated matrix

## Overview
This framework extends the  KKS model using the Phase-Field method to reproduce DSC tests. It serves as an open-source code realization of the research described in [the asoociated paper], focusing on [growth and coarseing of precipitates in LPBF AlSi10Mg alloy]. This implementation includes various computational tools and data to replicate and expand upon the paper's findings.

## related article 
Extension of a phase-field KKS model to predict the microstructure evolution in LPBF AlSi10Mg alloy submitted to non isothermal processes,
Seifallah Fetni, Jocelyn Delahaye, Héctor Sepúlveda , Laurent Duchêne, Anne Marie Habraken and Anne Mertens, Computational Materials Science
Volume 244, September 2024, 113197

[https://authors.elsevier.com/a/1jN7i_L4--JJYY](https://www.sciencedirect.com/science/article/abs/pii/S092702562400418X)

## Citation (if you find this study helpful)
@article{FETNI2024113197,
title = {Extension of a phase-field KKS model to predict the microstructure evolution in LPBF AlSi10Mg alloy submitted to non isothermal processes},
journal = {Computational Materials Science},
volume = {244},
pages = {113197},
year = {2024},
issn = {0927-0256},
doi = {https://doi.org/10.1016/j.commatsci.2024.113197},
author = {Seifallah Fetni and Jocelyn Delahaye and Héctor Sepúlveda and Laurent Duchêne and Anne Marie Habraken and Anne Mertens},
}

## Contributors
- **Code Developers**: [Jocelyn Delahaye and Seifallah El Fetni]
- Developped  code   in  ULiège  by  two  researchers  J.  Delahaye  PdD  thesis  (Delahaye, J. (2022). How to feed and validate a phase-field model predicting the evolution of microstructures and properties in AlSi10Mg processed by Selective Laser Melting [Doctoral thesis, ULiège -https://orbi.uliege.be/handle/2268/293324) and  S.  Fetni  (Post  Doc  Grant IN IPD-STEMA 2019 grant and faculty post doc grant 2021) article published  in Computational Materials Science 2024 (Extension of a KKS model to predict the microstructure evolution in LPBF AlSi10Mg alloy submitted to non isothermal processes  with doi: https://doi.org/10.1016/j.commatsci.2024.113197). 


## Code Structure
The project is organized into two main parts, focusing on different aspects of the Extended KKS model implementation, and consists of three primary Jupyter notebooks:

### Jupyter Notebooks
1. **Main Simulation Code**
   - `pyFi_origin.ipynb`: This notebook contains the core KKS code used for running the simulations. It is designed to handle the main computational processes including the evolution of microstructures.

2. **Post-Processing Tools**
   - `Postprocess.ipynb`: This notebook is utilized for the post-processing of simulation outputs. It deals with:
     - Mapping of microstructures.
     - Qualitative and Pre-quantitative analysis including microstructure evolution.

3. **DSC Analysis Tool**
   - `compute_DSC.ipynb`: Dedicated to computing quantitative data from simulations, specifically focusing on DSC heat curves.

### Additional Details
The so called "maps" store output data which includes various simulation details like:
- Concentrations of phases.
- Order parameters.
- Energy residuals.
These information are captured at regular intervals throughout the simulation process, enabling a thorough analysis of both the physical properties and the computational performance.

Note: The maps allow also the continuation/restart/bakcup of the simulation (checkpointing).

  

### Data and Scripts
- **Data Files**: 
  - `micro_final.npz`: Contains the Calphad data used within the simulations.
  - Initialization data are stored in a text file for easy access and manipulation (you can directly create/modify your initialization through the notebook of import it after external preparation).
- **Script Files**: 
  - Python script files (`*.py`) contain the various routines required to run the code. These scripts are automatically loaded by the main Jupyter notebook.

### Saved Outputs
The output maps are saved in associated folders, organized by:
- `big_grids/`: Contains maps used in the article for content representation.
- `small_grids/`: Contains maps for sensitivity analysis.
- `nucleation/`: Stores outputs related to the injection of nucleons within the initialization phase.

## Installation Requirements
- **Python Version**: Python 3.7 or more recent.
- 
### Dependencies
The code leverages modern Python libraries along with acceleration and parallelization packages to enhance performance. Key dependencies include:
- `numba`: A Just-In-Time compiler that translates a subset of Python and NumPy code into fast machine code.
- `pyfftw`: A Pythonic wrapper around the FFTW library, which is one of the fastest Fourier transform libraries available.
- Additionally, modern coding techniques such as vectorization and broadcasting are extensively employed to optimize computational efficiency. These techniques help to speed up operations by reducing the explicit loops in the code and making full use of the underlying hardware capabilities.


## Usage
To use this framework:
1. Ensure that you have Python 3.7 or higher installed along with the required libraries.
2. Clone this repository to your local machine.
3. Open the `pyFi_origin.ipynb` to run the KKS code or `Postprocess.ipynb` and `compute_DSC.ipynb` for post-processing tasks.

### Performance Recommendations
- **Regular Computing**: The framework can run on a standard desktop (e.g., 16 GB RAM, Intel i7 processor). This setup is adequate for handling smaller grids and less frequent mapping tasks. You can save your own maps and utilize them for various post-processing tasks and testing different input process parameters, such as temperature ramps and material properties.
- **High-Performance Computing (HPC)**: For larger simulations, such as those from 0.5 up to 1 µm squared, and for frequent mapping (saving of maps), using High-Performance Computing resources is recommended. HPC can significantly accelerate computation, especially for high-resolution and extensive simulation tasks.

### Reproducible run
For a complete and reproducible simulation setup, please download and extract [ReproducibleRun.zip], which contains all necessary scripts and files to run the simulation directly on your HPC or local machine.

### Customization
- Modify simulation parameters and inputs in the notebooks to explore different material properties (e.g. Calphad Data) and processing conditions (e.g. heat ramp). This flexibility allows for extensive customization depending on specific research needs or industrial applications.

## Licensing
You are free to use, edit, and distribute this code under [Apache License v2.0]. 

---

## Supplementary video:
- Simulation of the microsctructure evolution of AlSi10Mg SLM material during anisothermal heat load : matrix desaturation and growth and coalescence of d-Si precipitates. Further post-processing allows the obtention of the DSC curves
 as detailled in the related article and achieved by the post-process tool.
- ![](https://github.com/SFETNI/KKS-Phase-Field-code/blob/main/save_fig/Supp_Mat_computed_microstructure_evolution.gif)

For more information on how to setup and run the code, refer to the detailed comments within each script and notebook. If you encounter any issues or have questions, feel free to open an issue in this repository.
