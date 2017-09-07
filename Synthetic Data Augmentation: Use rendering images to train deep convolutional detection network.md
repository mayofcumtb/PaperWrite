# Synthetic Data Augmentation: Use rendering images to train deep convolutional detection network
## Abstract
Deep convolutional neural network has achieved great success on image recognition tasks. However, as a data-driven method, deep neural networks need many high quality labeled data. To overcome the shortage，a data augmentation technic has been proposed in this paper. The low-dimensional data are rendered from high-dimensional stereo data—3D CAD models which contain more appearance and geometry information. Meanwhile, detections and categories information are extracted in the rendering progress. Finally, the synthetic data as training sets and artificial labeled data as testing sets fetch into detection model for training which shows that synthetic data can help detection model achieve better performance.
## Introduction
Over the past decade, deep neural networks have significantly improved image classification[1,2] and object detection[3,4,5]. In contrast to other methods, deep neural networks require a mass of abundant labeled data in order to prevent over-fitting and enable complex networks to achieve better performance. However, compared to the image classification task, object detection label data is more expensive and laborious to be obtained。In view of the general phenomenon of insufficient markup data, many researchers have used different methods for data enhancement. We divided them into two categories: a) transformational images data augmentation method, b) synthetic images data augmentation method.
Transformational images data aumentation method convert images to another format images. Most common utitly of image data augmentaion is image rotation transform, flipping transform, shift transform, scale transform, contrast transform, noise transform and color transform.Rotation and reflection transform rotate image with angles, it can let a image become many images with one category.Shift transform move the  transform 
https://github.com/mayofcumtb/PaperWrite

> Compared with A, the image data commonly used means of rotating | enhanced reflection transformation (Rotation/reflection): random image rotation angle; to change the content of the image;
Flip transform (flip): flip images in horizontal or vertical directions;
Scaling transformation (zoom): zooming or zooming images in a certain proportion;
Translation transformation (shift): the image is translated in a certain way on the image plane;
The translation range and translation step size can be specified in a random or artificially defined manner, translating horizontally or vertically. Change the location of the image content;
Wavelet transform (scale): the image according to the scale factor specified, to enlarge or shrink; or according to the SIFT feature extraction method, using the scale factor of the specified image filtering structure of scale space. The image changes the size of the content or fuzzy degree;
Contrast transform (contrast): in the image of HSV color space, S and V change saturation luminance, keep the tone unchanged. H index operations for each pixel of the S and V components (from 0.25 to 4 between the factor index), increase the illumination change;
Noise perturbation (noise): random perturbation of RGB at each pixel of the image. The commonly used noise patterns are salt and pepper noise and Gauss noise;
Color transformation (color): the RGB color space of the pixel values of the training set is PCA, and the 3 principal direction vectors of the RGB space are obtained, 3 eigenvalues, P1, P2, P3, lambda 1, lambda 2, lambda 3.
## Reference
[1] A. Krizhevsky, I. Sutskever, and G.Hinton. ImageNet Classification with deep convolutional neural networks. In NIPS, 2012. 1,4,6
[2] Y. LeCun, B. Boser, J.Denker, D.Henderson, R. Howard, W.Hubbard, and L. Jackel. Backpropagation applied to handwritten zop code recognition.
[3] Ren, S., He, K., Girshick, R., & Sun, J. (2017). Faster r-cnn: towards real-time object detection with region proposal networks. IEEE Transactions on Pattern Analysis & Machine Intelligence, 39(6), 1137.
[4] Liu, W., Anguelov, D., Erhan, D., Szegedy, C., Reed, S., & Fu, C. Y., et al. (2016). SSD: Single Shot MultiBox Detector. European Conference on Computer Vision (pp.21-37). Springer, Cham.
[5] Redmon, J., & Farhadi, A. (2016). Yolo9000: better, faster, stronger
