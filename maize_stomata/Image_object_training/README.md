### Main question: Are morphorlogy of GC and SC are genotype-depend?

I previous used Random Classifier to classify genotypes of GC and SC using cell profiler 
extracted features. The accuracy is about 0.8. I want to understand if the accuracy is represent real biology (someGC and SC is phenotypically similar, which can be seen in confusion matrix. More details are in the CellProfiler_feature_analysis folder) or because of we did not capture enough features to distinguish different genotypes. 

I tried o use two methods to train raw masks/images to traing the classify genotyps.

### 1. PIXIMI
1. I developed the pipelone to crop the images into the objects and annotation files in Piximi supported 
Jason file.
2. Start to train the objects in Beta.Piximi. Training in larger datasets might need more stable version.

### 2. CNN model using keras and Tensorflow

I tried the build CNN from cratch with three layers: limited image numbers made the model overfitting and predictio accuracy was low. Droupout and data augumentation techniquewas were used, the overfitting was a litte bit improved, but the accuracy is still low (0.4) 

I fine-tued pre-trained model ResNet and VGG with the imageNet weights. The VGG had the best performance as accuracy is around 0.8
Hyperparameters are not tuned, but will do in the next step.
