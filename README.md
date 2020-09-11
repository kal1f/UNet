train.py: code to preprocess train images and train UNet

predict.py: code to preprocess test images, fit and predict masks. In the end it will plot image and predicted mask. Close image to see mask.

metric.py: eval metric (dice coef). You can add new metrics here as IoU. 

To find the nuclei in divergent images we will use UNet. It's powerful instrument in image segmantion. Our task is segmentation so task is not multiclass task(instance segmentation). So ou loss function will be becross_entropy. For evaluation we will use dice score.

First of all we wiil change images to smaller size to remove the load from the computer during processing.


We build Unet using Keras, it will give us to build model in easy way. To prevent overfitting we will use early stopping.
Too many epochs can lead to overfitting of the training dataset, whereas too few may result in an underfit model. Early stopping is a method that will allow us to specify an arbitrary large number of training epochs and stop training once the model performance stops improving on a hold out validation dataset. Also dropout can be used to prevent overfitting, but for me it gives nearly the same results. My best model got 0.87 dice score, I used batch_size = 32.



Moreover we should make sure that our eval function was implemented in right way. Values must be between 0 and  1. When we run train.py file we will see how change our loss function down and dice score up, so all is good. And when script finished our model we be save in model.h5 file, so you should'n run train everytime.


So, to try my code you should clone repo, create virtual env using requerments.txt (I used conda env) to install depedencies that I used. 

To install: 
1. git clone rep
2. cd rep
3. pip install venv
4. source venv/bin/activate
5. pip istall -r requerments.txt

Ro run:

1. python train.py 

2. python predict.py path/to/model

	example: python predict.py model.h5


A good way will be to add data as param to script in terminal, but I hardcoded it in script because this way is more easy for people who will try to use code first time. 