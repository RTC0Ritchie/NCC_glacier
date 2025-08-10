**Introduction to Code**

This code currently provides two main functionalities:
1. **Glacier Surface Velocity Extraction**: Utilizes cross-correlation techniques in the Fourier frequency domain (phase correlation) to extract glacier surface velocities based on two time-separated remote sensing images.
2. **Glacier Strain Rate Estimation**: Estimates the strain rate along a selected line over the glacier.

## Getting Started

Before using the code, you need to clone the entire repository to your local machine and install the required packages based on `requirements.txt`. Alternatively, you can use the `requirements.yml` file to create a conda environment automatically by running `conda env create -f requirements.yml`.

## Usage

A detailed usage guide based on case studies is provided in the user manual *Help.docx*. The manual uses the Helheim Glacier as an example to illustrate how to use the code to achieve the specified functions. The completion of the case study relies on the files in the `Example` folder. Users can create their own folders and follow the instructions in the manual to add the corresponding files as needed.

## Limitations

- Currently, only the Sentinel-2 series is supported.
- Only single-channel extraction is supported.
- Images need to be preprocessed and converted to JPG or TIF format before use.
- Data import (e.g., determination of extraction locations) still relies on pixel-level operations. Direct operations based on coordinates are under development.

We are grateful to *J. Deng* for providing us with ideas for improving the algorithm efficiency. URL: https://blog.csdn.net/djq_313/article/details/131178037

[![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

本作品采用
[知识共享署名-相同方式共享 4.0 国际许可协议][cc-by-sa]进行许可。

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg
