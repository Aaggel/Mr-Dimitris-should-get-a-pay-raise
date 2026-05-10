# Mr-Dimitris-should-get-a-pay-raise
Repository for my semester project on neural networks.

Project's scenario was the prediction of whether a person will have side-effects from a specific medication, solely based on their age and weight. Can be modified and improved as you see fit :)

Original prompt :
"Let's assume that an experimental drug was tested on people aged 13 to 100 in a clinical trial.
The trial involved 3000 participants. Half of the participants were under 65 years old, and the
other half were 65 years old and above.
The trial showed that approximately 90% of patients aged 65 and over experienced side
effects from the drug, while approximately 90% of patients under 65 did not experience side
effects.
Also, 10% of individuals over 65 who did not experience side effects were also under 70 kg,
generally indicating that elderly and overweight individuals were more likely to experience side
effects.
Additionally, 10% of patients who were under 65 and experienced side effects were over 70
kg, showing that younger overweight individuals were more likely to experience side effects.
We want to create a model that tells us whether a patient will experience side effects based
solely on their age and weight. The model's judgment will be based on training data generated
from random values for the two parameters, age, and weight.
Construct a neural network using TensorFlow - Keras. With two inputs, two outputs, and:
• 1st case, two hidden layers with 16 and 32 nodes respectively in each layer.
• 2nd case, three hidden layers with 16 32 64 nodes respectively in each layer.
For all the above, use appropriate activation functions that will lead to optimal results.
The input will accept the age and weight of the patient. When age<65, weight<70 and age>65,
weight<70, then the drug does not cause side effects, i.e., output 0. When age>65,
weight>=70, and age<=65, weight>=70, then the drug will cause side effects, output 1.
• Generate random values for the two parameters age and weight. Normalize the values
before passing through the neural network.
• Train your model first with 1500 and then with 3000 values.
• Use a learning rate of 0.01 and 0.0001 for each number of data.
• Use two different loss functions (MSE, sparse_categorical_crossentropy, etc.) for each
number of data.
• Use 20 and 30 epochs for each number of data.
• Use 10 and 50 batches for each number of data.
• Generate new random values for the two parameters age and weight to test your
neural network if it has been trained correctly.
Finally, visually represent your conclusions with a Confusion matrix or in any desired way for
each different case you performed above and record your observations. Explain
the code you created."

(I'm sorry if it looks weird, I don't know how to make the spaces show properly :'))
