from numpy import *


def compute_error_for_line_given_points(b, m, points):

    # initialize it at 0
    totalerror = 0
    # for every point
    for i in range(0, len(points)):
        # get the x value
        x = points[i, 0]

        # get the y value
        y = points[i, 1]

        # get the difference, square it and add it to the total
        totalerror += (y - (m*x + b)) ** 2

    # get the average of the total error
    return totalerror / float(len(points))


def gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations):
    # starting b and m
    b = initial_b
    m = initial_m

    # gradient descent
    for i in range(num_iterations):
        [b, m] = step_gradient(b, m, array(points), learning_rate)
    return [b, m]


def step_gradient(b_current, m_current, points, learningrate):
    # Starting points for our gradients
    b_gradient = 0
    m_gradient = 0

    n = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # direction with respect to b and m
        # computing partial derivatives with our error function
        b_gradient += -(2/n) * (y - (m_current*x + b_current))
        m_gradient += -(2/n) * x * (y - (m_current * x) + b_current)

    # update our b and m values using our partial derivatives
    new_b = b_current - (learningrate * b_gradient)
    new_m = m_current - (learningrate * m_gradient)
    return [new_b, new_m]


def run():
    # Step 1: Collect our data
    points = genfromtxt('../data/data.csv', delimiter=',')

    # Step 2: Define our hyperparameters
    # how fast our model should converge
    learning_rate = 0.0001
    # y = mx + b - Slope formula
    initial_b = 0
    initial_m = 0
    num_iterations = 1000

    # Step 3: Train our model
    print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m,
                                                                              compute_error_for_line_given_points(
                                                                                  initial_b, initial_m, points)))
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print("final gradient descent parameters: b = {1}, m = {2}, error = {3}".format(num_iterations, b, m,
                                                                                    compute_error_for_line_given_points(
                                                                                        b, m, points)))


if __name__ == '__main__':
    run()
