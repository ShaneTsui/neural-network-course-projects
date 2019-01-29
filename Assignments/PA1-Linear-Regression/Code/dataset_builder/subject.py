class Subject:
    def __init__(self):
        self.label_image_dict = {}

    def add(self, image, label):
        self.label_image_dict[label] = image

    def get(self, label):
        return self.label_image_dict[label]