# Public sensor data

The files in `raw/` are copies of the original public datasets. Experimental preprocessing, partitioning, and report classifiers are defined in `src/benchmark.py`.

## Occupancy Detection

Luis Candanedo (2016). *Occupancy Detection*. UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C5X01N.

Dataset page: https://archive.ics.uci.edu/dataset/357/occupancy+detection

Original download: https://archive.ics.uci.edu/static/public/357/occupancy+detection.zip

Original ZIP SHA-256: `4ae3f46aa98eedff564a9f6924d1635173e2fd2c816004342a9be93076d3a81a`

Associated study: Luis M. Candanedo and Véronique Feldheim (2016), “Accurate occupancy detection of an office room from light, temperature, humidity and CO2 measurements using statistical learning models,” *Energy and Buildings* 112, 28–39. https://doi.org/10.1016/j.enbuild.2015.11.071

## Room Occupancy Estimation

Adarsh Pal Singh and Sachin Chaudhari (2018). *Room Occupancy Estimation*. UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C5P605.

Dataset page: https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation

Original download: https://archive.ics.uci.edu/static/public/864/room+occupancy+estimation.zip

Original ZIP SHA-256: `a0db79ee96ea297c22458ac77a28fba20d41246e75d24b2655ca3ab624cc08fd`

Associated study: Adarsh Pal Singh, Vivek Jain, Sachin Chaudhari, Frank Alexander Kraemer, Stefan Werner, and Vishal Garg (2018), “Machine Learning-Based Occupancy Estimation Using Multivariate Sensor Nodes,” *2018 IEEE Globecom Workshops*, 1–6. https://doi.org/10.1109/GLOCOMW.2018.8644432

## Data license and transformations

Both datasets are distributed by UCI under Creative Commons Attribution 4.0: https://creativecommons.org/licenses/by/4.0/ . The original authors retain the data rights. The experiment derives binary sensor reports, binarizes the Room occupancy count, samples trace windows, and injects simulated faults. The repository's MIT license applies to its implementation.
