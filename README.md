#  Cough or Not detector 

Yuhua Jin,
Github Repository: [github repo with project work in](https://github.com/ucfninf/Cough-type-detector-CASA0018/blob/main/README.md) 
Edge Impulse Projects:[link to Edge Impulse projects](https://studio.edgeimpulse.com/studio/363571/create-impulse)


## Introduction
Early this spring, I came down with a severe case of the flu and had a persistent cough for over a month. This personal health challenge highlighted the importance of recognizing symptoms at an early stage outside of a traditional clinical setting, which inspired me to develop a tool to distinguish coughs from other sounds such as sneezes and ambient noise. The project involved creating a deep learning model by utilizing edge Impulse, that classifies audio inputs into three categories: coughs, sneezes, and background noises.
This sound-based model is specifically designed for the early detection of abnormalities associated with coughing, which can indicate a variety of health conditions. By accurately recognizing and classifying coughing sounds, the model aims to improve the accuracy of diagnosis in environments such as hospitals and clinics, thereby assisting healthcare professionals. The overall goal of the project is not only to refine the capabilities of the deep learning technology, but also to optimize its use in public health infrastructure. The goal of the project aims to reduce the risk of infection, support medical decision-making, and ultimately provide a tool to enhance early diagnosis and maintain vigilance in public health surveillance.
![image](https://github.com/ucfninf/Cough-type-detector-CASA0018/assets/146268411/d00fdfff-cf67-485f-a670-ea38c92f5bab) (Verywell Health, 2023)


## Research Question
How can a deep learning model, trained using Edge Impulse, effectively differentiate between cough sounds, sneezes, and background noise based on their spectral and temporal characteristics?

## Application Overview
This project integrates a sound classification system utilizing a deep learning model trained through Edge Impulse, designed to differentiate between cough sounds, sneezes, and background noise. The project consists of the following three components.
1.Data Collection: The first component involves gathering a diverse set of audio samples, which include distinct coughs, sneezes, and various background noises. These samples are recorded under different settings to simulate real-world conditions, ensuring the model learns to recognize sounds accurately across various environments.
2.Model Training and Validation: Using Edge Impulse, these sound clips are processed to extract relevant features, primarily focusing on spectral and temporal characteristics. After this step, a convolutional neural network (CNN) is trained with these features, undergoing several iterations of training and validation to optimize accuracy and minimize overfitting.
3.Model Deployment: Once trained, the model is converted into a TensorFlow Lite format, making it suitable for deployment on low-power devices like the Arduino Nano BLE 33. This conversion ensures that the model can run efficiently in real-time applications.
Data from the deployed system can be used to further refine the model, enhancing its ability to adapt to new and changing environments, thus improving its diagnostic accuracy and utility in public health. 
 
The Following is the flow chart about the application workflows.

![image](https://github.com/ucfninf/Cough-type-detector-CASA0018/assets/146268411/3c4b646a-8595-41a4-a308-bdd15b8f39de)

## Data
Collection:
At the outset of the project, approximately 500 cough sound samples were sourced from open-source platforms and academic institutions, encompassing a variety of cough types including dry, wet, and those from COVID-19 positive patients. To augment the diversity of the dataset, additional recordings were made with the help of classmates.
Originally, the project aimed to categorize coughs into detailed types: dry, wet, and pseudo-coughs. However, this approach led to model overfitting issues， the model is hard to classify the specific cough categories. Consequently, all cough sounds were consolidated into a single generalized category. Moreover, to broaden the range of audio inputs for model training, sounds of sneezes and background noises from medical settings were incorporated into the dataset.
During the data collection phase, I also utilized resources from the Coswara project—an initiative that has made available manual annotations of crowdsourced COVID-19 coughs. As of March 2021, the Coswara dataset included 1,500 samples. (Sharma, N. et al. 2021)   These samples, particularly categorized into 'positive' and 'negative' COVID-19 coughs, were selectively integrated into this Edge Impulse model to enhance the representativeness and robustness of the training dataset.

![image](https://github.com/ucfninf/Cough-type-detector-CASA0018/assets/146268411/8c774a07-59af-4d5c-a782-f3c8a9039313)

Preprocessing and Data Handling
To labeling data and preprocessing initially, the audio files were labeled as 'dry cough' or 'wet cough', including data collected offline from classmates. Additionally, labels for 'fake cough (dry cough 1 labeled in the image below)' were introduced to account for variations in the recordings. Each sound file was meticulously trimmed and uniformly distributed across the training and testing datasets. To enhance the robustness of the dataset against real-world conditions, artificial noise was added, and all files were normalized to ensure a consistent sampling rate. Files were also adjusted in length to maintain uniformity.
Observations and Adjustments:
After the initial preprocessing and during the early rounds of model training, it became apparent that all types of coughs exhibited highly similar spectral features. This similarity led to overfitting, with the model achieving a maximum accuracy of only 69.2%. In response, the dataset underwent a significant restructuring: all cough sounds were amalgamated into a single category, the 'fake cough' labels were removed, and additional audio types such as sneezes and background noises from medical environments were incorporated. This reclassification and the standardization of audio lengths to a consistent duration improved model accuracy considerably.
![image](https://github.com/ucfninf/Cough-type-detector-CASA0018/assets/146268411/c614d310-8c1f-40a6-bbe9-b8019ff1b9e7)
![image](https://github.com/ucfninf/Cough-type-detector-CASA0018/assets/146268411/893b2767-1cb1-413c-b8e2-bad6daeabb1e)

Outcome on Data Classification:

![image](https://github.com/ucfninf/Cough-type-detector-CASA0018/assets/146268411/6847c3b6-0b39-4be1-a2e5-8155f7eba4ae)

Following these adjustments, the audio samples were categorized into clear groups: Cough, Sneeze, and Background noise, enabling more straightforward and effective training. The final dataset was further refined to balance the dataset, resulting in 339 training and 92 testing sets of combined 'cough and non-cough' audio clips. Each segment was meticulously curated to ensure clarity and uniformity, ultimately standardizing the final dataset to consist of 12-minute audio clips for each category.

## Model
Based on the intention of the project and its data. This Project has chosen 3 kind of Processing Block (MFCC, MFE, Spectrogram) and 1 learning block (Classification)

Data Characteristics Analysis:

![image](https://github.com/ucfninf/Cough-type-detector-CASA0018/assets/146268411/0e3cf6a9-876c-4699-8a6b-9ea01a3a81f3)

The model experimented with MFCCs, Mel Frequency Energy (MFE), and spectrograms. Initially, the project utilized MFE module, which achieved a training accuracy of 100%. The capacity of MFE to provide a clear energy disStribution makes it particularly useful for identifying the distinct energy signatures of sounds like coughs, which have unique energy profiles. However, given that the model needed to classify sounds based on frequency without relying on linguistic content, and to handle the complexity added by continuous background noise, spectrograms proved more fitting. Their visual representation of sound frequencies over time aligns well with Convolutional Neural Networks (CNNs), which are adept at processing image data. The focus of this project is to focus on the temporal dynamics of sounds, such as the rapid onset and end of a cough. 

![image](https://github.com/ucfninf/Cough-type-detector-CASA0018/assets/146268411/41629f3b-f2d8-4325-9e5d-bd52dd507b7e)

On the other hand, spectrograms also present challenges; they can complicate the model's processing tasks and require substantial computational resources to manage their high-dimensional data. Nevertheless, the ability of spectrograms to capture detailed time-frequency dynamics has made them the best choice for this project, leading to their selection for further model development. This approach has significantly enhanced the model’s accuracy and its ability to classify audio events within varied acoustic environments. 

![image](https://github.com/ucfninf/Cough-type-detector-CASA0018/assets/146268411/aeb35c6e-2b3c-4e05-a831-a36261ce2af8)

## Experiments
Throughout the experiment, to accurately grasp the specific impact of different settings on the performance of the model, it is necessary to adjust a series of key parameters in a targeted manner. These adjustments cover learning rate, number of training cycles, layer structure reshaping, number of neurons, and dropout ratio. The purpose of this is to deeply compare the model training accuracy, test accuracy, and loss under different parameter configurations, and then determine the optimal parameter combination.

The following is a partial statistic of all the experiments I have done. The following records only include experimental models with all parameters and accuracy higher than 50% and successfully built.

Training Records
![image](https://github.com/ucfninf/Cough-type-detector-CASA0018/assets/146268411/c7b2e9d2-9d52-43b9-a262-f014815b5e95)

I have first tried MFE and Classification based on the three categorizations (Cough, Sneeze, Background Noise). Then I moved to Spectrogram. I conducted a series of parameter optimization experiments for classifiers based on Mel Frequency Cepstral Coefficients (MFE) and spectrograms. These experiments focused on the adjustment of key parameters to assess their impact on model performance.
For the MFE model, the initial experiment showed that the model achieved 100% training accuracy and a mere 0.02 training loss, indicating that the model adapted exceptionally well to the training data. 
![image](https://github.com/ucfninf/Cough-type-detector-CASA0018/assets/146268411/978617e6-95af-4ffa-bdee-8a6fe7a7a34a)![image](https://github.com/ucfninf/Cough-type-detector-CASA0018/assets/146268411/8f8dcfb2-fa29-4c93-a1c2-0730952aa177)![image](https://github.com/ucfninf/Cough-type-detector-CASA0018/assets/146268411/63db8c48-2a06-4d19-bad0-f250279f7299)

In subsequent experiments, I explored the effects of different configurations by adjusting the learning rate (from 0.005 to 0.001), the training cycles (fixed at 100), and the Dropout rate (from 0.25 to 0.1). These adjustments helped the model perform better in preventing overfitting and improving generalization.

![image](https://github.com/ucfninf/Cough-type-detector-CASA0018/assets/146268411/296f3cfa-e84a-4d88-bbe1-34f3b8bcd3b1)![image](https://github.com/ucfninf/Cough-type-detector-CASA0018/assets/146268411/7f3132bc-c2ad-495e-a2ec-f7d3846b175f)

Similarly, I conducted experiments on the spectrogram classifier, particularly adjusting the window size. For example, setting the frame length to 0.001 seconds, the frame stride to 0.01 seconds, and the FFT length to 128 significantly improved the model's accuracy and stability. After adjusting the module with these settings, the accuracy rate increased significantly and tended to stabilize. By gradually reducing the learning rate from 0.005 to 0.0001, and then to 0.0008 in Experiment 5, the model's test accuracy became more stable, indicating that a lower learning rate helps the model adjust weights more finely, thereby enhancing generalization.

After I exported the Spectrogram model for the first time, I found that the deploy was unsuccessful and I needed to adjust the window size: After adjusting the module, the accuracy increased significantly and tended to be stable.

![image](https://github.com/ucfninf/Cough-type-detector-CASA0018/assets/146268411/a55abd12-e11e-45c3-b3ae-92b8df285db5)
After adjusting the module, the accuracy rate has increased significantly and tends to be stable


## Results and Observations
Experiment 9 and 15 are the best choices for Spectrogram and MFE models. These experiments demonstrated optimal performance and provided the foundation for further practical applications, highlighting their potential in real-world settings.
![image](https://github.com/ucfninf/Cough-type-detector-CASA0018/assets/146268411/d2db7159-e4ca-4915-b92c-20c1a5503690)![image](https://github.com/ucfninf/Cough-type-detector-CASA0018/assets/146268411/03d33ca9-dd0e-4155-990a-b3f9395a196a)
Model Successfully deployed Experiment 9 on Arduino Nano BLE 33. 

Ultimately, after selecting the best-performing model, I successfully deployed it on the Arduino Nano BLE 33 Sense. However, the deployment process was not without challenges, such as recording failures, highlighting the necessity for continuous adjustments and optimizations.
In terms of deployment, I uploaded the trained model to an Arduino BLE 33 Sensor to assess its functionality in a real-world environment. The initial deployment utilized training data that had previously yielded reliable results. Despite this, the performance did not meet expectations due to the initially oversized window settings, prompting a critical reassessment of the most suitable threshold settings before continuing with the deployment on the Arduino Nano BLE Sense. This step was essential to ensure the model’s effectiveness and reliability in practical applications.

With more time, I would continue to refine the deep learning model and conduct further tests until achieving optimized results. I would also introduce more environmental noise and sound sources to enhance the model's accuracy. Moreover, I plan to conduct field deployments in hospitals or other medical facilities to validate the project's value and explore potential applications in various fields. These efforts would improve the model’s robustness and adaptability in diverse and challenging environments.


## Bibliography
1.	Edge Impulse Documentation. (2023). Run inference on Arduino. Edge Impulse Documentation. [Online]. Available: https://docs.edgeimpulse.com/docs/run-inference/arduino-library
2.	Sharma, N. et al. (2021). Data repository of Project Coswara. GitHub. [Online]. Available: https://github.com/iiscleap/Coswara-Data
3.	Ali, S. N., & Shuvo, S. B. (2021). Hospital Ambient Noise Dataset. Kaggle. [Data set]. Available: https://doi.org/10.34740/KAGGLE/DSV/2173743
4.	Google Research. (2021). AudioSet - Sneeze. Google Research. [Online]. Available: https://research.google.com/audioset/eval/sneeze.html
5.	YouTube. (2023). Step-by-step tutorial on tinML application using edge impulse and Arduino - environment installation python node.js CLI installation and precautions. [Video]. Available: https://www.youtube.com/watch?v=aoOXic7JmOw
6.	Arduino Documentation. (2022). Microphone sensor on Arduino Nano 33 BLE Sense. Arduino Documentation. [Online]. Available: https://docs.arduino.cc/tutorials/nano-33-ble-sense/microphone-sensor/
7.	Orlandic, L., Teijeiro, T. & Atienza, D. The COUGHVID crowdsourcing dataset, a corpus for the study of large-scale cough analysis algorithms. Sci Data 8, 156 (2021). https://doi.org/10.1038/s41597-021-00937-4
8.	Andrewmvd. (2020). COVID-19 Cough Audio Classification [online] Kaggle. Available at: https://www.kaggle.com/datasets/andrewmvd/covid19-cough-audio-classification [Accessed 23 April 2024].
9.	Sarabhian, (2022). Coswara Dataset (Heavy Cough) [online] Kaggle. Available at: https://www.kaggle.com/datasets/sarabhian/coswara-dataset-heavy-cough [Accessed 1 April 2024].
10.	Nafin, (2021). Hospital Ambient Noise [online] Kaggle. Available at: https://www.kaggle.com/datasets/nafin59/hospital-ambient-noise?resource=download [Accessed 03 April 2024].
11.	Verywell Health, (2023). Persistent Cough: Causes, Evaluation, and Treatment [image online] Available at: https://www.verywellhealth.com/persistent-cough-causes-evaluation-2249305 [Accessed 23 April 2024].

----

## Declaration of Authorship

I, Yuhua Jin, confirm that the work presented in this assessment is my own. Where information has been derived from other sources, I confirm that this has been indicated in the work.


Yuhua Jin

April 25th, 2024
