from src.HAR import ActivityRecognizer

x = ActivityRecognizer()
x.preprocess_data(sequence_length = 1024)
x.generate_model()
x.train_model()
x.evaluate_model()