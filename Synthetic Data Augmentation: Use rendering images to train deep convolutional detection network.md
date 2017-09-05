# Synthetic Data Augmentation: Use rendering images to train deep convolutional detection network
## Abstract
Deep convolutional neural network has achieved great success on image recognition tasks. However, as a data-driven method, deep neural networks need many high quality labeled data. To overcome the shortage，a data augmentation technic has been proposed in this paper. The low-dimensional data are rendered from high-dimensional stereo data—3D CAD models which contain more appearance and geometry information. Meanwhile, detections and categories information are extracted in the rendering progress. Finally, the synthetic data as training sets and artificial labeled data as testing sets fetch into detection model for training which shows that synthetic data can help detection model achieve better performance.
## Introduction
Over the past decade, deep neural networks have significantly improved image classification[1,2] and object detection[3,4,5]. In contrast to other methods, deep neural networks require a mass of abundant labeled data in order to prevent over-fitting and enable complex networks to achieve better performance. However, compared to the image classification task, object detection label data is more expensive and laborious to be obtained。In view of the general phenomenon of insufficient markup data, many researchers have used different methods for data enhancement. We divided them into two categories: a) transformational images data augmentation method, b) synthetic images data augmentation method.
Transformational images data aumentation method convert images to another format images, 

## Reference
[1] A. Krizhevsky, I. Sutskever, and G.Hinton. ImageNet Classification with deep convolutional neural networks. In NIPS, 2012. 1,4,6
[2] Y. LeCun, B. Boser, J.Denker, D.Henderson, R. Howard, W.Hubbard, and L. Jackel. Backpropagation applied to handwritten zop code recognition.
[3] Ren, S., He, K., Girshick, R., & Sun, J. (2017). Faster r-cnn: towards real-time object detection with region proposal networks. IEEE Transactions on Pattern Analysis & Machine Intelligence, 39(6), 1137.
[4] Liu, W., Anguelov, D., Erhan, D., Szegedy, C., Reed, S., & Fu, C. Y., et al. (2016). SSD: Single Shot MultiBox Detector. European Conference on Computer Vision (pp.21-37). Springer, Cham.
[5] Redmon, J., & Farhadi, A. (2016). Yolo9000: better, faster, stronger.
