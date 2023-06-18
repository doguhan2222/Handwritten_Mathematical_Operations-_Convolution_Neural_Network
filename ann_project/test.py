from alexnet import AlexNet
from ourCnnModel import OurCnnModel
import tensorflow as tf

def main():

    # AlexNet Model

    # cnn_model = AlexNet()
    # cnn_model.load_dataset()
    # cnn_model.print_dataset()
    # cnn_model.visualizing_dataset()
    # cnn_model.preprocessing_data()
    # cnn_model.defining_model()
    # model = cnn_model.alexNet()
    # print(cnn_model.model_summary(model))
    # hist = cnn_model.training_model(model)
    # cnn_model.loss_and_accuracy_plot(hist)
    # reports = cnn_model.classification_report(model)
    # cnn_model.cnfs_matrix(reports)
    # cnn_model.saving_model(model)

    #OurCnnModel Model

    # new_model = OurCnnModel()
    # new_model.load_dataset()
    # new_model.print_dataset()
    # new_model.visualizing_dataset()
    # new_model.preprocessing_data()
    # new_model.defining_model()
    # model = new_model.our_model()
    # print(new_model.model_summary(model))
    # hist = new_model.training_model(model)
    # new_model.loss_and_accuracy_plot(hist)
    # reports = new_model.classification_report(model)
    #
    # new_model.cnfs_matrix(reports)
    # new_model.saving_model(model)

    # Test of AlexNet on saving model

    # alexnet_model = tf.keras.models.load_model('saved_model/alexNet_ann_hw.h5')
    # image_path = 'ourtest/deneme3.jpg'
    # AlexNet.test_equation(image_path, alexnet_model)

    # Test of OurModel on saving model

    # our_test_model = tf.keras.models.load_model('saved_model/our_model_ann_hw.h5')
    # image_path = 'ourtest/deneme3.jpg'
    # OurCnnModel.test_equation(image_path, our_test_model)


if __name__ == "__main__":
    main()
