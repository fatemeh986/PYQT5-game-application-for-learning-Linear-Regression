import matplotlib.pyplot as plt
import numpy as np

def generate_values(a, b, x_values, noise_type='gaussian', noise_scale=5):
    # Generate y values without noise
    y_values = [a * x + b for x in x_values]

    # Add noise based on the specified type
    if noise_type == 'gaussian':
        noise = np.random.normal(0, noise_scale, len(x_values))
    elif noise_type == 'uniform':
        noise = np.random.uniform(-noise_scale, noise_scale, len(x_values))
    else:
        raise ValueError("Invalid noise type. Choose 'gaussian' or 'uniform'.")

    # Add noise to y values
    y_values_with_noise = [y + n for y, n in zip(y_values, noise)]

    return y_values_with_noise, noise

def main():
    # Task 1: Generate y = ax + b values based on parameters
    a = 2.5
    b = 20
    x_values = list(range(-10, 11))

    # Task 2: Create noise: gaussian or uniform
    y_values_with_gaussian_noise, gaussian_noise = generate_values(a, b, x_values, noise_type='gaussian', noise_scale=5)
    y_values_with_uniform_noise, uniform_noise = generate_values(a, b, x_values, noise_type='uniform', noise_scale=5)

    # Task 3: Take the previous array and add noise to it
    y_values, _ = generate_values(a, b, x_values, noise_type=None, noise_scale=0)

    # Task 4: Create a histogram based on the generated noise
    plt.figure(figsize=(12, 5))

    # Histogram for Gaussian noise
    plt.subplot(1, 2, 1)
    plt.hist(gaussian_noise, bins=20, color='red', alpha=0.7)
    plt.title('Histogram of Gaussian Noise')
    plt.xlabel('Noise Values')
    plt.ylabel('Frequency')

    # Histogram for Uniform noise
    plt.subplot(1, 2, 2)
    plt.hist(uniform_noise, bins=20, color='green', alpha=0.7)
    plt.title('Histogram of Uniform Noise')
    plt.xlabel('Noise Values')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    # Task 5: Calculate prediction and errors
    predictions = [a * x + b for x in x_values]
    predictions_with_gaussian_noise = [a * x + b for x in x_values]
    predictions_with_uniform_noise = [a * x + b for x in x_values]

    errors = np.array(y_values) - np.array(predictions)
    errors_with_gaussian_noise = np.array(y_values_with_gaussian_noise) - np.array(predictions_with_gaussian_noise)
    errors_with_uniform_noise = np.array(y_values_with_uniform_noise) - np.array(predictions_with_uniform_noise)

    # Plot the errors
    plt.figure(figsize=(12, 5))

    # Plot the errors without noise
    plt.subplot(1, 3, 1)
    plt.plot(x_values, errors, 'bo-', label='Errors without noise')
    plt.xlabel('x')
    plt.ylabel('Error')
    plt.title('Errors without Noise')
    plt.legend()
    plt.grid(True)

    # Plot the errors with Gaussian noise
    plt.subplot(1, 3, 2)
    plt.plot(x_values, errors_with_gaussian_noise, 'ro-', label='Errors with Gaussian noise')
    plt.xlabel('x')
    plt.ylabel('Error')
    plt.title('Errors with Gaussian Noise')
    plt.legend()
    plt.grid(True)

    # Plot the errors with Uniform noise
    plt.subplot(1, 3, 3)
    plt.plot(x_values, errors_with_uniform_noise, 'go-', label='Errors with Uniform noise')
    plt.xlabel('x')
    plt.ylabel('Error')
    plt.title('Errors with Uniform Noise')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
