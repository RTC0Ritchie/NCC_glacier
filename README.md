**Introduction to Code**

The provided code implements a Python-based image matching solution using the Fast Fourier Transform (FFT) and Normalized Cross Correlation (NCC) techniques to extract the **glacier surface velocity**. It is designed to perform precise displacement estimation between two images.

The code focuses on processing grayscale images (single band) with a defined search window and range. It employs a two-stage search strategy to extract glacier surface velocity. The initial rapid search focuses on efficiently obtaining highly reliable results. A subsequent refined search is then conducted to fill in the gaps left by the first search.

Two core classes are included:

*img2vxy_fourier*: Implements FFT-based image correlation for matching template images within a larger image.

*img2vxy_equal*: Provides methods for filling in unmatched points in the first search process using surrounding displacement information.

The original file supports the input of *jpg* format images. The pre_process file can quickly convert the *Sentinel-2A/B* files downloaded by the European Space Agency into the format supported by NCC code.

The code is structured to facilitate easy modification and extension by changing other classes and functions. Researchers can adjust parameters like kernel size and window size to suit their specific application needs. The modular design also allows for customization of image reading and preprocessing steps.

The repository dataset and the sample file *statistics.xlsx* provided serves as a demonstration of FFT-based image matching techniques. Researchers are encouraged to adapt the code to their specific image datasets and application requirements. The Python dependencies required for running the code are listed in the *requirements.txt* file.

We are grateful to *J. Deng* for providing us with ideas for improving the algorithm efficiency. URL: https://blog.csdn.net/djq_313/article/details/131178037

[![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

本作品采用
[知识共享署名-相同方式共享 4.0 国际许可协议][cc-by-sa]进行许可。

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg
