import tensorflow as tf
import random
import numpy as np

class Adversarial(object):
    # Function to create adversarial pattern
    def adversarial_pattern(self, model, image, label):
        image = tf.cast(image, tf.float32)
    
        with tf.GradientTape() as tape:
            tape.watch(image)
            prediction = model(image)
            loss = tf.keras.losses.MSE(label, prediction)
    
        gradient = tape.gradient(loss, image)
    
        signed_grad = tf.sign(gradient)
    
        return signed_grad

    def create_adversarial_example(self, model, img_rows, img_cols, image, channels, image_label, epsilon=0.1):
        perturbations = self.adversarial_pattern(model, image.reshape((1, img_rows, img_cols, channels)), image_label).numpy()
        adversarial = image + perturbations * epsilon
        return adversarial

    def generate_adversarials(self, model, x_train, y_train, img_rows, img_cols, channels, batch_size):
        while True:
            x = []
            y = []
            for batch in range(batch_size):
                N = random.randint(0, 100)

                label = y_train[N]
                image = x_train[N]
            
                perturbations = self.adversarial_pattern(model, image.reshape((1, img_rows, img_cols, channels)), label).numpy()
            
            
                epsilon = 0.1
                adversarial = image + perturbations * epsilon
            
                x.append(adversarial)
                y.append(y_train[N])
        
        
            x = np.asarray(x).reshape((batch_size, img_rows, img_cols, channels))
            y = np.asarray(y)
        
            yield x, y