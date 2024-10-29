# polylogs

This repository provides functionality revolving around the computation, simplification and integration of symbols of multiple polylogarithms. For details, refer to the PDFs thesis.pdf (which provides general theory as well as details regarding the predecessor project which can als be found on my github) and symbology.pdf.

Dependencies: \
**C++ (13+)** \
 | gcc and g++ compiler \
 | GNU make, GNU autoconf, GNU automake, GNU libtool, GNU sed \
 | GiNaC \
 |-- CLN \
 |---- GNU MP (*you might want to configure CLN such that it uses the GMP library*) \
 | fplll \
 |-- GNU MPFR \
 | LinBox \
 |-- Givaro \
 |-- fflas-ffpack \
 |---- OpenBLAS

 

**Wolfram Mathematica** \
 | PolyLogTools \
 |-- HPL

 Note: dependencies are listed where they occur for the first time. In particular, working right to left and top to bottom sequentially should work.

**N.B.**: The project is currently unfinished and on hold due to my studies at Cambridge. Some functionality is missing and refactoring is necessary. It has been developed between October 2023 and October 2024 during my time as a student research assistant in Prof. Lorenzo Tancredi's group at TUM.
