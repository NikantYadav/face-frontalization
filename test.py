import numpy as np
from keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
from datagenerator import DataGenerator, DataGenerator_predict
from glob import glob

# Define paths to the dataset
dataset_dir = "dataset"  # Replace this with the actual path to your dataset

# parameters
height = 128
width = 128
channels = 3
z_dimension = 512
batch_size = 64
epochs = 10000
line = 3
n_show_image = 1

# Data generator - will automatically handle loading side and frontal images based on folder structure
datagenerator = DataGenerator(dataset_dir, batch_size=batch_size)  # Train data generator
datagenerator_p = DataGenerator_predict(dataset_dir, batch_size=batch_size)  # Prediction data generator

# Optimizers
optimizerD = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999)
optimizerG = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999)


def conv2d_block(layers, filters, kernel_size=(4, 4), strides=2, momentum=0.8, alpha=0.2):
    input = layers
    layer = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same")(input)
    layer = BatchNormalization(momentum=momentum)(layer)
    output = LeakyReLU(alpha=alpha)(layer)
    return output


def Conv2DTranspose_block(layers, filters, kernel_size=(4, 4), strides=2, momentum=0.8, alpha=0.2):
    input = layers
    layer = Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(input)
    layer = BatchNormalization(momentum=momentum)(layer)
    output = LeakyReLU(alpha=alpha)(layer)
    return output


class Gan():
    def __init__(self):
        self.height = height
        self.width = width
        self.channels = channels
        self.z_dimension = z_dimension
        self.batch_size = batch_size
        self.optimizerD = optimizerD
        self.optimizerG = optimizerG
        self.number = 0

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizerD, metrics=['accuracy'])

        self.generator = self.build_generator()

        self.discriminator.trainable = False

        z = Input(shape=(self.height, self.width, self.channels))
        image = self.generator(z)

        valid = self.discriminator(image)

        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizerG)
    

    def build_discriminator(self):
        input = Input(shape=(self.height, self.width, self.channels))
        layers = conv2d_block(input, 16)
        layers = conv2d_block(layers, 32)
        layers = conv2d_block(layers, 64)
        layers = conv2d_block(layers, 128)
        layers = conv2d_block(layers, 256)
        layers = conv2d_block(layers, 512)
        output = Conv2D(1, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='sigmoid')(layers)

        model = Model(input, output)
        model.summary()
        return model

    def build_generator(self):
        input = Input(shape=(self.height, self.width, self.channels))
        layers = conv2d_block(input, 16)
        layers = conv2d_block(layers, 32)
        layers = conv2d_block(layers, 64)
        layers = conv2d_block(layers, 128)
        layers = conv2d_block(layers, 256)
        layers = conv2d_block(layers, 512)
        layers = conv2d_block(layers, 512)

        layers = Conv2DTranspose_block(layers, 512)
        layers = Conv2DTranspose_block(layers, 256)
        layers = Conv2DTranspose_block(layers, 128)
        layers = Conv2DTranspose_block(layers, 64)
        layers = Conv2DTranspose_block(layers, 32)
        layers = Conv2DTranspose_block(layers, 16)
        output = Conv2DTranspose(filters=3, kernel_size=(4, 4), strides=2, activation='tanh', padding='same')(layers)

        model = Model(input, output)
        return model

    def build_combined_model(self):
        self.discriminator.trainable = False
        z = Input(shape=(self.height, self.width, self.channels))
        image = self.generator(z)
        valid = self.discriminator(image)

        return Model(z, valid)

    def train(self, epochs, batch_size, save_interval):
        #discriminator_output_shape = self.discriminator.output_shape[1:]
        #fake = np.zeros((batch_size, *discriminator_output_shape))
        #real = np.ones((batch_size, *discriminator_output_shape))
        fake = np.zeros((batch_size, 1, 1, 1))
        real = np.ones((batch_size, 1, 1, 1))

        print(f"Discriminator output shape: {self.discriminator.output_shape}")

        for epoch in range(epochs):
            for batch in range(datagenerator.__len__()):
                # Get real and fake images
                side_images, front_images = datagenerator.__getitem__(batch)

                # Debugging: Check the shapes of the input batches
                print(f"Epoch {epoch + 1}/{epochs}, Batch {batch + 1}/{datagenerator.__len__()}")
                print(f"Shape of side_images: {side_images.shape}, Shape of front_images: {front_images.shape}")

                # Train discriminator
                generated_images = self.generator.predict(side_images)
                
                # Debugging: Check the shape of generated images
                print(f"Shape of generated images: {generated_images.shape}")
                self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizerD, metrics=['accuracy'])

                discriminator_fake_loss = self.discriminator.train_on_batch(generated_images, fake)
                discriminator_real_loss = self.discriminator.train_on_batch(front_images, real)
                discriminator_loss = (discriminator_fake_loss[0] + discriminator_real_loss[0]) * 0.5


                #discriminator_loss = np.add(discriminator_fake_loss, discriminator_real_loss) * 0.5
                
                # Debugging: Check losses for discriminator
                print(f"Discriminator fake loss: {discriminator_fake_loss}, Discriminator real loss: {discriminator_real_loss}")
                print(f"Total Discriminator Loss: {discriminator_loss}")

                # Train generator
                generator_loss = self.combined.train_on_batch(side_images, real)

                # Debugging: Check generator loss
                print(f"Generator Loss: {generator_loss}")

            if epoch % save_interval == 0:
                # Save images and model after each epoch
                save_path = f'generated/Training{line}/'
                self.save_image(epoch, batch, front_images, side_images, save_path)

                predict_side_images = datagenerator_p.__getitem__(0)
                save_path = f'generated/Predict{line}/'
                self.save_predict_image(epoch, batch, predict_side_images, save_path)
                self.generator.save(f"generated/Predict{line}/{line}_{epoch}.h5")

            datagenerator.on_epoch_end()
            datagenerator_p.on_epoch_end()

    def save_image(self, epoch, batch, front_image, side_image, save_path):
        generated_image = (0.5 * self.generator.predict(side_image) + 0.5)
        front_image = (255 * ((front_image) + 1) / 2).astype(np.uint8)
        side_image = (255 * ((side_image) + 1) / 2).astype(np.uint8)

        for i in range(self.batch_size):
            plt.figure(figsize=(8, 2))
            plt.subplots_adjust(wspace=0.6)

            # Plot generated image
            generated_image_plot = plt.subplot(1, 3, 1)
            generated_image_plot.set_title('Generated Image')
            plt.imshow(generated_image[i])

            # Plot original front image
            original_front_face_image_plot = plt.subplot(1, 3, 2)
            original_front_face_image_plot.set_title('Original Front Image')
            plt.imshow(front_image[i])

            # Plot original side image
            original_side_face_image_plot = plt.subplot(1, 3, 3)
            original_side_face_image_plot.set_title('Original Side Image')
            plt.imshow(side_image[i])

            for ax in [generated_image_plot, original_front_face_image_plot, original_side_face_image_plot]:
                ax.axis('off')

            save_name = f'{epoch}-{batch}-{i}.png'
            save_path = os.path.join(save_path, save_name)

            if not os.path.isdir(save_path):
                os.makedirs(save_path)

            plt.savefig(save_name)
            plt.close()

    def save_predict_image(self, epoch, batch, side_image, save_path):
        generated_image = 0.5 * self.generator.predict(side_image) + 0.5
        side_image = (255 * ((side_image) + 1) / 2).astype(np.uint8)

        for i in range(self.batch_size):
            plt.figure(figsize=(8, 2))
            plt.subplots_adjust(wspace=0.6)

            generated_image_plot = plt.subplot(1, 2, 1)
            generated_image_plot.set_title('Generated Image')
            plt.imshow(generated_image[i])

            original_side_face_image_plot = plt.subplot(1, 2, 2)
            original_side_face_image_plot.set_title('Original Side Image')
            plt.imshow(side_image[i])

            for ax in [generated_image_plot, original_side_face_image_plot]:
                ax.axis('off')

            save_name = f'{epoch}-{batch}-{i}.png'
            save_path = os.path.join(save_path, save_name)

            if not os.path.isdir(save_path):
                os.makedirs(save_path)

            plt.savefig(save_name)
            plt.close()


if __name__ == '__main__':
    dcgan = Gan()
    dcgan.train(epochs=epochs, batch_size=batch_size, save_interval=n_show_image)
