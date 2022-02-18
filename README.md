# Towards-Best-Practice-of-Interpreting-Deep-Learning-Models-for-EEG-based-BCI
Understanding deep learning models is important for EEG-based brain-computer interface (BCI), since it not only can boost trust of end users but also potentially shed light on reasons that cause a model to fail. However, deep learning interpretability has not yet raised wide attention in this field. It remains unknown how reliably existing interpretation techniques can be used and to which extent they can reflect the model decisions. 

In order to fill this research gap, we conduct the first quantitative evaluation and explore the best practice of interpreting deep learning models designed for EEG-based BCI. We design metrics and test seven well-known interpretation techniques on benchmark deep learning models. We propose a set of processing steps that allow the interpretation results to be visualized in an understandable and trusted way.

In this project, we implemented 7 interpretation techniques on two benchmark deep learning models "EEGNet" and "InterpretableCNN" (10.1109/TNNLS.2022.3147208) for EEG-based BCI. The methods include:
    
gradient×input, 
DeepLIFT, 
integrated gradient, 
layer-wise relevance propagation (LRP),
saliency map, 
deconvolution,g
guided backpropagation

The dataset for the test is available from 
https://figshare.com/articles/dataset/EEG_driver_drowsiness_dataset/14273687

Details of our work can be found in the paper 
"Towards Best Practice of Interpreting Deep Learning Models for EEG-based Brain Computer Interfaces"
https://arxiv.org/abs/2202.06948

If you find the codes useful, pls cite our paper.

If you have met any problems, you can contact Dr. Cui Jian at cuij0006@ntu.edu.sg



