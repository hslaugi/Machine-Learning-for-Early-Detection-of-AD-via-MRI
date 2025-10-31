# Machine-Learning-for-Early-Detection-of-AD-via-MRI
Class Project for Medical Imaging

**Transfer Learning-based Deep Learning Method for Early Detection of Alzheimer’s Disease from MRI Scans**

**Background:** Alzheimer’s disease (AD) is a progressive neurodegenerative disorder characterized by cognitive decline and structural brain changes. MRI scans can capture early alterations such as hippocampal atrophy, providing valuable diagnostic information.
Recent advancements in artificial intelligence (AI) have enabled automated medical image analysis through deep learning. Transfer learning allows pretrained convolutional neural networks (CNNs) such as ResNet50 to be adapted for disease classification tasks with limited data.
Objective: Build a classification model based on transfer learning technique that distinguishes four classes of Alzheimer’s Disease (AD) subjects using MRI brain scans from the Alzheimer’s Disease OASIS MRI dataset. The four classes are: Very Mild Dementia, Mild Dementia, Moderate Dementia, and No Dementia.

**Dataset: **
The dataset we will use is the OASIS MRI dataset (https://sites.wustl.edu/oasisbrains/), which consists of 80,000 cross-sectional brain MRI images from 416 subjects aged 18 to 96. The images have been divided into four classes based on Alzheimer's progression. These four classes are: Very Mild Dementia, Mild Dementia, Moderate Dementia, and No Dementia. The number of images for each class are 5002, 488, 67000, and 13700, respectively. The dataset aims to provide a valuable resource for analyzing and detecting early signs of Alzheimer's disease.

**Models: **
We will try four types of models for classification, namely ResNet-50 [1], vgg-19 [2], Transformer[3], and Inception-v3 [5]. All the models will be pretrained on relevant image datasets. Then the trained models will be fine-tuned on the OASIS MRI dataset for AD classification. 
Expected results: 
1.	The Accuracy of four class classifications reach acceptable ranges (>80%) in the OASIS MRI dataset. 
2.	Transfer learning has higher accuracy than the model without pretrained weights initialization. 
3.	Model structure that yields the highest accuracy will be found.
   
**Skills & Tools Gained by the end of this project:**
•	Core ML concepts: supervised learning, data preprocessing, model evaluation.
•	Deep learning: CNN architecture, feature extraction, transfer learning.
•	Practical experience with TensorFlow, Keras, and Python data analysis libraries.
•	Exposure to AI workflows used in medical imaging research.

**Reference**
[1] Kaiming He et al. Deep Residual Learning for Image Recognition. CVPR.
[2] S. Liu and W. Deng, "Very deep convolutional neural network based image classification using small training sample size," 2015 3rd IAPR Asian Conference on Pattern Recognition (ACPR), Kuala Lumpur, Malaysia, 2015, pp. 730-734, doi: 10.1109/ACPR.2015.7486599. 
[3] Ashish Vaswani et al. Attention Is All You Need. Google inc. 
[4] Christian Szegedy. Rethinking the Inception Architecture for Computer Vision. Google inc. 
[5] Szegedy, Christian, et al. "Rethinking the inception architecture for computer vision." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
