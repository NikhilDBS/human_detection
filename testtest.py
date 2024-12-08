import numpy as np
import cv2
import os
from preprocessor import process_image
from main import loadModelFile, predict, sobel, compute_gradient_magnitude_angle, calc_hog, calc_lbp  # Import necessary functions from main.py 
def calculateFeatureVectorImg_HOG(img_path):
    """
    @param 1: img_path, full path of the image
    @return feature_vector, contains features which is used as an input to neural network. dimension [7524 x 1]
    """
    img_c = cv2.imread(img_path)
    img_gray_scale = np.round(0.299 * img_c[ :, :, 2 ] + 0.587 * img_c[ :, :, 1 ] + 0.114 * img_c[ :, :,0 ])  
    gx, gy = sobel(img_gray_scale)
    gradient_magnitude, gradient_angle = compute_gradient_magnitude_angle(gx, gy)

    # Removed code by US
    # img_path = img_path.split('/')

    # save gradient magnitude files for test images.
    # if "Test_" in img_path[ 1 ]:
    #     if not os.path.exists("Gradient Magnitude Test Images"):
    #         os.makedirs("Gradient Magnitude Test Images")
    #     cv2.imwrite("Gradient Magnitude Test Images" + "/" + str(img_path[ 2 ]), gradient_magnitude)

    feature_vector = calc_hog(img_gray_scale, gradient_magnitude,
                              gradient_angle)  # calculate hog descriptior

    feature_vector2 = calc_lbp(img_gray_scale)
    #feature_vector = feature_vector.reshape(feature_vector.shape[ 0 ],1) 
    # Removed by Author
    # reshaping vector. making dimension [7524 x 1]
    # this below code is used to store the feature vector of crop001278a.bmp and crop001278a.bmp into txt file.
    # feature_vector2 = feature_vector2.reshape(feature_vector2.shape[0], 1)
    # # print(feature_vector.shape,feature_vector2.shape)
    # feature_vector1 = np.append(feature_vector,feature_vector2)
    # feature_vector1 = feature_vector1.reshape(feature_vector1.shape[0], 1)

    # Removed code by US
    # if img_path[ 2 ] == "crop001034b.bmp":
    #     if not os.path.exists("HOG descriptor"):
    #         os.makedirs("HOG descriptor")

    #     # saving hog descriptor value. Here,%10.14f will store upto 14 decimal of value
    #     np.savetxt("HOG descriptor" + "/" + str(img_path[ 2 ][ :-3 ]) + "txt", feature_vector, fmt="%10.14f")
    # np.savetxt("HOG-LBP descriptor" + "/" + str(img_path[2][:-3]) + "txt", feature_vector1, fmt="%10.14f")
    # np.savetxt("LBP descriptor" + "/" + str(img_path[2][:-3]) + "txt", feature_vector2, fmt="%10.14f")
    return feature_vector



def main(image_path, model_file):
    # Load the trained model
    model = loadModelFile(model_file)

    # Calculate the feature vector for the custom image using HOG
    feature_vector = calculateFeatureVectorImg_HOG(image_path)

    # Reshape the feature vector to match the input shape expected by the model
    feature_vector = feature_vector.reshape(feature_vector.shape[0], 1)


    # Make a prediction using the trained model
    prediction = predict(feature_vector, model)

    # Classify the prediction
    classification = 1 if prediction[0][0] > 0.5 else 0

    # Print the result
    print(f"Prediction for the image '{image_path}': {'Human detected (1)' if classification == 1 else 'No human detected (0)'}")

if __name__ == "__main__":
    # Specify the path to your custom image and the model file
    input_path = "custom test images/silhouette-of-person-peeking-into-dark-room-through-royalty-free-image-1686773134.jpg"
    process_image(input_path)
    custom_image_path = "test.bmp"  # Change this to your image path
    model_file_name = "data_hog400"  # Change this to the appropriate model file name with .npy extension

    main(custom_image_path, model_file_name)