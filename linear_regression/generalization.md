# Generalization
The fundimental problem of machine learning and statistics is that we want to make sure that our model has truly discoverd a *general* pattern rather than simply memorizing the data.

The phenomenon of fitting closer to our training data than to the general pattern is called *overfitting*, and techniques for combatting it are called *regularization* methods

## Training Error vs Generalization Error

In standard supervised learning, we assume that the training data and the test data are drawn *independently* from *identical* distributions. 
This assumbtion called *the IID assumbtion* is important as it allows us to estimate the performance of a model on new data.

Under the IID assumbtion, we can define two different kinds of error. *Training Error*, which measures how well the model performs on the specific examples it was trained on. 
And *generalization error*, which measures how well the model performs on the entire data distribution that produced both the training and test sets.

While the training error is straightforward to compute, the generalization error is not easily computable because we do not know the true distribution of all real world data.

So to approximate the generalization error, we evaluate the model on a seperate test set that was not used during training. This test set acts as the *"unseen"* data and gives an estimate of how well the model is expected to perform in practice.

## Underfitting or Overfitting
When comparing traning and test erors, we want to be mindful of two common situation.

First, when our training error and validation error are both high and close to each other, the model is not able to fit the training data well,
this could mean that our model is too simple to capture the pattern that we are trying to model. This is known as *underfitting*

On the other hand, when the training error is significantly lower than our validation error, indicating severe *overfitting*, meaning the model has learned the training set too closely and does not generalize as well.

>Note that overfitting is not always a bad thing. In deep learning especially, the best predictive models often perform far better on training data than on the test data.

What matters is how low we can push the validation error, not whether the gap exists. If the training error reaches zero, the gap becomes exactly equal to the generalization error, and any improvement must come from reducing that gap.

![OverFitting vs Underfitting](./imgs/overfittingvsunderfitting.png)


## Dataset Size

Another big consideration to bear in mind is dataset size. As we increase the amount of training data, the generalization error typically decreases. 


## Model Selection 

The next question is how to choose the model that performs best in practice. Different model can vary in their architectures and hyperparameters.
 Since each of these can influence performance, we need a systematic way to compare them.

A key rule is that no information from the test set should influence this process. 
The test set is meant to give an unbiased estimate of how well the final chosen model will generalize. 
If we use it during model selection, we risk tuning our model to the specific quirks of the test data rather than learning patterns that generalize.

Relying on training data alone is also not enough because training error cannot estimate generalization. 

This is why a seperate validation set is introduced. 
The validation set guides decision such as hyperparameter tuning and architecture selection, while the test set is kept untouched until the very end.

### Cross Validation

When training data is too small, we might not be able to afford to hold out enough data for a validation set, *Cross-Validation* becomes a practical alternative. 

It basically splits the original data into $K$ non-overlapping substs.
Then model training and validation are executed $K$ times, each time training on $K-1$ subsets and validating on a different subset.
Finally, The training and validation erros are estimated by averaging over the results from $K$ experiments.
