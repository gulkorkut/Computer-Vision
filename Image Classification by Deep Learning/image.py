import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.applications import VGG16, ResNet50, InceptionV3
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import load_img, img_to_array
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def load_and_preprocess_data(dataset_path):
    apple_types = os.listdir(dataset_path)
    image_paths = []
    labels = []
    
    for apple_type in apple_types:
        apple_type_path = os.path.join(dataset_path, apple_type)
        images = os.listdir(apple_type_path)
        
        for image in images:
            image_path = os.path.join(apple_type_path, image)
            image_paths.append(image_path)
            labels.append(apple_type)

    label_to_int = {label: i for i, label in enumerate(apple_types)}
    labels = [label_to_int[label] for label in labels]

    train_image_paths, test_image_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42
    )

    def load_images(image_paths, target_size=(224, 224)):
        images = []
        for path in image_paths:
            img = load_img(path, target_size=target_size)
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
        return np.array(images)

    X_train = load_images(train_image_paths)
    X_test = load_images(test_image_paths)

    y_train = np.array(train_labels)
    y_test = np.array(test_labels)

    return X_train, X_test, y_train, y_test, apple_types

def build_custom_model(base_model, classes, model_name):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(len(classes), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions, name=model_name)

    for layer in base_model.layers[:-20]:
        layer.trainable = True

    model.compile(optimizer=Adam(lr=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f'Test accuracy for {model.name}: {test_acc}')

def train_and_evaluate_classifiers(classifiers, X_train_flat, y_train, model_name):
    for classifier in classifiers:
        scores = cross_val_score(classifier, X_train_flat, y_train, cv=5)
        print(f'{classifier.__class__.__name__} accuracy for {model_name}: {np.mean(scores)}')

def extract_surf_features(images, descriptor_length=32):
    orb = cv2.ORB_create()
    orb_features = []
    
    for img in images:
        kp, des = orb.detectAndCompute(img, None)
        
        if des is not None:
            des = des[:descriptor_length]
            des = np.pad(des, ((0, max(0, descriptor_length - len(des))), (0, 0)), 'constant')
            orb_features.append(des)

    if len(orb_features) > 0:
        return np.concatenate(orb_features, axis=0)
    else:
        return np.array([])

def extract_orb_features(images):
    orb = cv2.ORB_create()
    orb_features = []
    
    for img in images:
        kp, des = orb.detectAndCompute(img, None)
        orb_features.append(des)

    return np.vstack(orb_features)

def main():
    dataset_path = '/kaggle/input/apple-datasets'
    X_train, X_test, y_train, y_test, apple_types = load_and_preprocess_data(dataset_path)

    base_model_vgg16 = VGG16(include_top=False, weights='imagenet')
    base_model_resnet50 = ResNet50(include_top=False, weights='imagenet')
    base_model_inceptionv3 = InceptionV3(include_top=False, weights='imagenet')

    model_vgg16 = build_custom_model(base_model_vgg16, apple_types, 'VGG16_Model')
    model_resnet50 = build_custom_model(base_model_resnet50, apple_types, 'ResNet50_Model')
    model_inceptionv3 = build_custom_model(base_model_inceptionv3, apple_types, 'InceptionV3_Model')

    for model in [model_vgg16, model_resnet50, model_inceptionv3]:
        train_and_evaluate_model(model, X_train, y_train, X_test, y_test)

    X_train_custom, X_test_custom, y_train_custom, y_test_custom = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    X_train_surf, X_test_surf, y_train_surf, y_test_surf = train_test_split(
        X_train_custom, y_train_custom, test_size=0.2, random_state=42
    )
    X_train_orb, X_test_orb, y_train_orb, y_test_orb = train_test_split(
        X_train_custom, y_train_custom, test_size=0.2, random_state=42
    )

    X_train_surf_features = extract_surf_features(X_train_surf, descriptor_length=32)
    X_train_orb_features = extract_orb_features(X_train_orb)

    classifiers = [
        SVC(),
        MLPClassifier(),
        KNeighborsClassifier(),
        DecisionTreeClassifier(),
        RandomForestClassifier()
    ]

    train_and_evaluate_classifiers(classifiers, X_train_surf_features, y_train_surf, 'SURF')
    train_and_evaluate_classifiers(classifiers, X_train_orb_features, y_train_orb, 'ORB')

if __name__ == "__main__":
    main()