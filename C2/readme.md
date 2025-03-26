# Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization



### key points for Initialization taken from assignment
- The weights  𝑊[𝑙] should be initialized randomly to break symmetry.
- However, it's okay to initialize the biases  𝑏[𝑙] to zeros. Symmetry is still
broken so long as  𝑊[𝑙] is initialized randomly.

**Symmetry Breaking Vs Zero Initialization:**
- https://community.deeplearning.ai/t/symmetry-breaking-versus-zero-initialization/16061

**Observations**:
- The cost starts very high. This is because with large random-valued weights, the last activation (sigmoid) outputs results that are very close to 0 or 1 for some examples, and when it gets that example wrong it incurs a very high loss for that example. Indeed, when  log(𝑎[3])=log(0), the loss goes to infinity.
- Poor initialization can lead to vanishing/exploding gradients, which also slows down the optimization algorithm.
- If you train this network longer you will see better results, but initializing with overly large random numbers slows down the optimization.

- Initializing weights to very large random values doesn't work well.
- Initializing with small random values should do better. The important question is,
how small should be these random values be? Let's find out up next!

**OptionalRead**
```
The main difference between Gaussian variable (numpy.random.randn()) and uniform random variable is the distribution of the generated random numbers:

numpy.random.rand() produces numbers in a uniform distribution.
and numpy.random.randn() produces numbers in a normal distribution.
When used for weight initialization, randn() helps most the weights to Avoid being close to the extremes, allocating most of them in the center of the range.

An intuitive way to see it is, for example, if you take the sigmoid() activation function.

You’ll remember that the slope near 0 or near 1 is extremely small, so the weights near those extremes will converge much more slowly to the solution, and having most of them near the center will speed the convergence.
```

- Finally, try "He Initialization"; this is named for the first author of He et al., 2015. (If you have heard of "Xavier initialization", this is similar except Xavier initialization uses a scaling factor for the weights  𝑊[𝑙]
of sqrt(1./layers_dims[l-1]) where He initialization would use sqrt(2./layers_dims[l-1]).)

**takeaways**
- Different initializations lead to very different results
- Random initialization is used to break symmetry and make sure different hidden units can learn different things
- Resist initializing to values that are too large!
- He initialization works well for networks with ReLU activations


### key points for Regularization taken from assignment
**L2-regularization Observations:**
    - The value of  𝜆 is a hyperparameter that you can tune using a dev set.
    - L2 regularization makes your decision boundary smoother. If  𝜆 is too large, it is also possible to "oversmooth",    resulting in a model with high bias.

**What is L2-regularization actually doing?:**
    - L2-regularization relies on the assumption that a model with small weights is simpler than a model with large weights. Thus, by penalizing the square values of the weights in the cost function you drive all the weights to smaller values. It becomes too costly for the cost to have large weights! This leads to a smoother model in which the output changes more slowly as the input changes.

**What you should remember: the implications of L2-regularization on:**
- The cost computation:
    - A regularization term is added to the cost.
- The backpropagation function:
    - There are extra terms in the gradients with respect to weight matrices.
- Weights end up smaller ("weight decay"):
    - Weights are pushed to smaller values.

**Dropout**

- The idea behind drop-out is that at each iteration, you train a different model that uses only a subset of your neurons. With dropout, your neurons thus become less sensitive to the activation of one other specific neuron, because that other neuron might be shut down at any time.

- Inverted dropout explanation???? (understand better)
    - Set  𝐴[1] to  𝐴[1]∗𝐷[1]. (You are shutting down some neurons). You can think of  𝐷[1] as a mask, so that when it is multiplied with another matrix, it shuts down some of the values.

    - Divide  𝐴[1] by keep_prob. By doing this you are assuring that the result of the cost will still have the same expected value as without drop-out. (This technique is also called inverted dropout.)

    - You had previously shut down some neurons during forward propagation, by applying a mask  𝐷[1] to A1. In backpropagation, you will have to shut down the same neurons, by reapplying the same mask  𝐷[1] to dA1.

    - During forward propagation, you had divided A1 by keep_prob. In backpropagation, you'll therefore have to divide dA1 by keep_prob again (the calculus interpretation is that if  𝐴[1] is scaled by keep_prob, then its derivative  𝑑𝐴[1] is also scaled by the same keep_prob).

- Note:
  - A common mistake when using dropout is to use it both in training and testing. You should use dropout (randomly eliminate nodes) only in training. Deep learning frameworks like TensorFlow, PaddlePaddle, Keras or caffe come with a dropout layer implementation. Don't stress - you will soon learn some of these frameworks.
  
- What you should remember about dropout:
    - Dropout is a regularization technique.
    - You only use dropout during training. Don't use dropout (randomly eliminate nodes) during test time.
    - Apply dropout both during forward and backward propagation.
    - During training time, divide each dropout layer by keep_prob to keep the same expected value for the activations. For example, if keep_prob is 0.5, then we will on average shut down half the nodes, so the output will be scaled by 0.5 since only the remaining half are contributing to the solution. Dividing by 0.5 is equivalent to multiplying by 2. Hence, the output now has the same expected value. You can check that this works even when keep_prob is other values than 0.5.


### key points for Gradient Checking taken from assignment

**Notes**
    - Gradient Checking is slow! Approximating the gradient with  ∂𝐽/∂𝜃 ≈ 𝐽(𝜃+𝜀)−𝐽(𝜃−𝜀)/2𝜀 is computationally costly. For this reason, we don't run gradient checking at every iteration during training. Just a few times to check if the gradient is correct.
    - Gradient Checking, at least as we've presented it, doesn't work with dropout. You would usually run the gradient check algorithm without dropout to make sure your backprop is correct, then add dropout.

**What you should remember from this notebook:**
    - Gradient checking verifies closeness between the gradients from backpropagation and the numerical approximation of the gradient (computed using forward propagation).
    - Gradient checking is slow, so you don't want to run it in every iteration of training. You would usually run it only to make sure your code is correct, then turn it off and use backprop for the actual learning process.


### Optimization Algos

**note on creating mini-batches:**
    - Shuffling and Partitioning are the two steps required to build mini-batches
    - Powers of two are often chosen to be the mini-batch size, e.g., 16, 32, 64, 128.

**GD with momentum**
- The velocity is initialized with zeros. So the algorithm will take a few iterations to "build up" velocity and start to take bigger steps.
- If  𝛽=0 , then this just becomes standard gradient descent without momentum.

How do you choose  𝛽?
    - The larger the momentum  𝛽 is, the smoother the update, because it takes the past gradients into account more. - - But if  𝛽 is too big, it could also smooth out the updates too much.
    - Common values for  𝛽 range from 0.8 to 0.999. If you don't feel inclined to tune this,  𝛽=0.9 is often a reasonable default.
    - Tuning the optimal  𝛽 for your model might require trying several values to see what works best in terms of reducing the value of the cost function  𝐽.
    
**What you should remember:**
- Momentum takes past gradients into account to smooth out the steps of gradient descent. It can be applied with batch gradient descent, mini-batch gradient descent or stochastic gradient descent.
- You have to tune a momentum hyperparameter  𝛽 and a learning rate  𝛼.

