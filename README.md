# Bidirectional-RNN

## Preprocessing
1. The images is resized to 28*28. 
2. After this, the images is converted to black and white images using simple thresholding technique.
3. The pixel values are normalized for faster computation. 
4. Labels are converted into one-hot encoding vector for the classification problem.

## Model
1. Output layer of the architecture uses Softmax activation unit.
2. The error metric is softmax cross-entropy. The main reason for using softmax cross-entropy was because the gradient of the SCE error is directly proportional to the update made at that point.
3. Adam optimizer is used for faster convergence.

## Results
Recognition Rate: 90.9%
