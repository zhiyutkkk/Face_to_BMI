# Face_to_BMI

Body Mass Index is an important indicator of a person’s health. Traditionally, BMI is
calculated by measuring the height and weight of a person, which takes manual effort. Deep
learning neural networks can help automate this task by predicting a person’s BMI with one’s
facial features. Previous work had used many approaches including computer vision, deep
learning, and computational. Since the dataset I have is very small, I adopted the transfer
learning approach. I used three pre-trained models including VGG16, ResNet-50, and SENet-50,
and finetuned them on the VisualBMI dataset. The ResNet-50 model showed the best
performance with an MAE score of 7.5. The model was deployed in Streamlit so that users can
upload images or use a webcam for BMI prediction.
