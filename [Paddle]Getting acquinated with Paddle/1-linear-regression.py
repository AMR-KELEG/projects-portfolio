import numpy as np
import paddle.fluid as fluid

if __name__ == '__main__':
    '''
    A simple script for training a linear regression model on a toy dataset.
    Input:
    X_train: One dimensional data points
    y_train: The correct output
    
    Aim:
    Find w,b such that y_pred = w*x + b is as close as possible to y_train
    
    Takeaways:
    * data is similar to tensorflow's placeholder
    * Executor is similar to tensorflow's session
    * You can define the model in a way similar to pytorch
    * Passing X_train is equivalent to treating the dataset as a single batch
    '''
    
    # Load the dataset
    X_train = np.array([1., 2., 3., 4.]).astype('float32').reshape(-1, 1)
    y_train = np.array([2., 4., 6., 8.]).astype('float32').reshape(-1, 1)

    # Define the input and label variables
    x = fluid.layers.data(name="x", shape=[1], dtype='float32')
    y = fluid.layers.data(name="y", shape=[1], dtype='float32')
    
    # Define the model
    y_pred = fluid.layers.fc(input=x, size=1, act=None)
    
    # Define the cost function and optimizer
    cost = fluid.layers.square_error_cost(input=y_pred, label=y)
    avg_cost = fluid.layers.mean(cost)

    # Setting learning rate to 1 instead of the default 0.2 seemed better
    opt = fluid.optimizer.Adam(learning_rate=1.0)
    opt.minimize(avg_cost)

    # Define the executor
    cpu = fluid.core.CPUPlace()
    exe = fluid.Executor(cpu)
    
    # Initialize the variables
    exe.run(fluid.default_startup_program())

    # Train the model
    for _ in range(100):
        # Since cost and avg_cost are defined then the whole variables (x,y) must be fed to the executor.
        # i.e.: The line below won't execute since y is missing
        # pred = exe.run(feed={'x': X_train}, fetch_list=[y_pred.name])
        pred = exe.run(feed={'x': X_train, 'y': y_train}, fetch_list=[y_pred.name])
    
    # Print the final predictions for the Training points
    print(pred)
