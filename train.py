from ultralytics import YOLO

# Load a model
model = YOLO("yolov8s.yaml")  # build a new model from scratch
model = YOLO("yolov8s.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="dataset.yaml", epochs=10, patience=5, batch=128)  # train the model
# model.save("/valohai/outputs/models/trained.pt")  # save trained model to trained.pt
