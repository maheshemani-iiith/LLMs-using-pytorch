"""
PyTorch Activation Functions - Implementation Guide
====================================================

CONCEPT: Activation functions introduce non-linearity to neural networks.
Without them, stacked linear layers would just be one big linear transformation.

WHY DO WE NEED THEM?
- Linear combinations of linear functions = still linear (can't learn complex patterns)
- Non-linearity allows networks to learn curves, decision boundaries, etc.

KEY PROPERTIES:
1. Differentiable (for backpropagation)
2. Fast to compute
3. Produces gradients that don't vanish/explode
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# 1. RELU (Rectified Linear Unit)
# ============================================================================
# Formula: ReLU(x) = max(0, x)
# Output range: [0, ∞)
# Use case: Hidden layers in most modern networks
# Pros: Simple, fast, avoids vanishing gradient
# Cons: "Dying ReLU" - dead neurons outputting only zeros

class ReLU_Manual(nn.Module):
    """Implement ReLU: max(0, x)

    Hint: Use torch.clamp() or torch.where()
    """
    def forward(self, x):
        return torch.clamp(x, min=0)


class ReLU_BuiltIn(nn.Module):
    """Use PyTorch's built-in ReLU"""
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x)
        

x = torch.tensor([-2, -1, 0, 1, 2, 3], dtype=torch.float32)
relu = ReLU_Manual()
print(relu(x))

# ============================================================================
# 2. SIGMOID - Classical S-shaped Function
# ============================================================================
# Formula: Sigmoid(x) = 1 / (1 + e^(-x))
# Output range: (0, 1) - useful for binary classification
# Use case: Output layer for binary classification, gates in LSTM
# Pros: Produces probabilities
# Cons: Vanishing gradients, slower training

class Sigmoid_Manual(nn.Module):
    """Implement Sigmoid: e^(x) / (1 + e^(x))

    Hint: Use torch.exp()
    """
    def forward(self, x):
        return torch.exp(x) / (1 + torch.exp(x))



class Sigmoid_BuiltIn(nn.Module):
    """Use PyTorch's numerically stable sigmoid"""
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid(x)
        


# ============================================================================
# 3. TANH (Hyperbolic Tangent)
# ============================================================================
# Formula: Tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
# Output range: (-1, 1) - zero-centered, better than sigmoid
# Use case: Hidden layers, RNNs
# Pros: Zero-centered output, stronger gradients than sigmoid
# Cons: Still suffers from vanishing gradients

class Tanh_Manual(nn.Module):
    """Implement Tanh using exponentials

    Hint: Formula above, use torch.exp()
    """
    def forward(self, x):
        return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))


class Tanh_BuiltIn(nn.Module):
    """Use PyTorch's built-in Tanh"""
    def __init__(self):
        super().__init__()
        self.tanh = nn.Tanh()

    def forward(self, x):
        return self.tanh(x)


# ============================================================================
# 4. LEAKY RELU - Improved ReLU
# ============================================================================
# Formula: LeakyReLU(x) = x if x > 0, else alpha * x (alpha is small, e.g., 0.01)
# Output range: (-∞, ∞) but slopes are small for x < 0
# Use case: When ReLU causes dying neurons
# Pros: Allows negative gradients, no dying neurons
# Cons: Adds hyperparameter alpha

class LeakyReLU_Manual(nn.Module):
    """Implement LeakyReLU with learnable alpha

    Hint: Use torch.where() to choose between x and alpha*x
    """
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return torch.where(x > 0, x, self.alpha*x)
        


# ============================================================================
# 5. ELU (Exponential Linear Unit)
# ============================================================================
# Formula: ELU(x) = x if x > 0, else alpha * (e^x - 1)
# Output range: (-alpha, ∞)
# Use case: Alternative to ReLU/LeakyReLU
# Pros: Smooth for negative values, mean closer to 0
# Cons: More expensive (exponential)

class ELU_Manual(nn.Module):
    """Implement ELU

    Hint: torch.where() for branching, torch.exp() for negative side
    """
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return torch.where(x>0, x, self.alpha * (torch.exp(x)-1))


# ============================================================================
# 6. GELU (Gaussian Error Linear Unit)
# ============================================================================
# Formula: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
# Output range: Similar to x, but scaled
# Use case: Transformers (BERT, GPT, etc.)
# Pros: Smooth, better than ReLU in transformers
# Cons: More computationally expensive

class GELU_Manual(nn.Module):
    """Implement GELU using approximation formula

    Hint: Break it down:
    1. Calculate x³
    2. Calculate the tanh argument
    3. Apply tanh
    4. Multiply by x and 0.5
    """
    def forward(self, x):
        x_cubed = x ** 3
        tanh_arg = torch.sqrt(torch.tensor(2.0 / np.pi)) * (x + 0.044715 * x_cubed)
        return 0.5 * x * (1 + torch.tanh(tanh_arg))
        


class GELU_Exact(nn.Module):
    """Implement exact GELU using error function

    Formula: x * 0.5 * (1 + erf(x / √2))
    Hint: Use torch.erf()
    """
    def forward(self, x):
        return torch.gelu(x)
        


# ============================================================================
# 7. SOFTMAX - Probability Distribution
# ============================================================================
# Formula: Softmax(x_i) = e^(x_i) / Σ(e^(x_j))
# Output range: (0, 1), sums to 1 - probability distribution
# Use case: Output layer for multi-class classification
# Pros: Natural probability interpretation
# Important: Use numerically stable version (subtract max)

class Softmax_Manual(nn.Module):
    """Implement numerically stable Softmax

    Hint: Subtract max(x) before computing exp to avoid overflow
    Division: exp(x - max) / sum(exp(x - max))
    """
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        max_d = torch.max(x, dim = self.dim, keepdim=True)[0] # tensor with max values 
        adj_x = x - max_d
        exp_x = torch.exp(adj_x)
        sum_exp = torch.sum(exp_x, dim=self.dim, keepdim=True)
        return exp_x / sum_exp 

class Softmax_BuiltIn(nn.Module):
    """Use PyTorch's built-in Softmax"""
    def __init__(self, dim=1):
        super().__init__()
        self.softmax = nn.Softmax(dim=dim)

    def forward(self, x):
        return self.softmax(x)


# ============================================================================
# 8. SOFTPLUS - Smooth ReLU Alternative
# ============================================================================
# Formula: Softplus(x) = ln(1 + e^x)
# Output range: (0, ∞)
# Use case: Positive outputs needed (e.g., variance in Gaussian)
# Pros: Differentiable everywhere, smooth
# Cons: Similar to ReLU but more expensive

class Softplus_Manual(nn.Module):
    """Implement Softplus: ln(1 + e^x)

    Hint: Use torch.log() and torch.exp()
    """
    def forward(self, x):
        return torch.log(1+torch.exp(x))

class Softplus_BuiltIn(nn.Module):
    """Use PyTorch's built-in Softplus"""
    def __init__(self):
        super().__init__()
        self.softplus = nn.Softplus()

    def forward(self, x):
        return self.softplus(x)

# ============================================================================
# 9. SWISH (Self-Gated Activation)
# ============================================================================
# Formula: Swish(x) = x * Sigmoid(x)
# Output range: Similar to input but smoothed
# Use case: Modern architectures, EfficientNets
# Pros: Self-gating mechanism, often better than ReLU
# Cons: More expensive than ReLU

class Swish_Manual(nn.Module):
    """Implement Swish: x * Sigmoid(x)

    Hint: Combine x with sigmoid output
    """
    def forward(self, x):
        return x*torch.sigmoid(x)


class Swish_BuiltIn(nn.Module):
    """Swish with learnable scaling: x * Sigmoid(beta * x)

    Hint: Create a learnable parameter beta using nn.Parameter()
    """
    def __init__(self):
        super().__init__()
        self.beta = beta = nn.Parameter(torch.tensor(1.0))


    def forward(self, x):
        return x*torch.sigmoid(self.beta * x)


# ============================================================================
# 10. MISH - Self-Regularizing Activation
# ============================================================================
# Formula: Mish(x) = x * Tanh(Softplus(x))
# Output range: Similar to input
# Use case: Deep networks, offers good empirical results
# Pros: Self-regularizing, smooth
# Cons: Computational cost

class Mish_Manual(nn.Module):
    """Implement Mish: x * Tanh(Softplus(x))

    Hint: Use F.softplus() then torch.tanh()
    """
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class Mish_BuiltIn(nn.Module):
    """Use PyTorch's built-in Mish"""
    def __init__(self):
        super().__init__()
        self.mish = nn.Mish()

    def forward(self, x):
        return self.mish(x)
# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def test_activation(activation, x, name="Activation"):
    """Helper to test an activation function"""
    y = activation(x)
    print(f"{name}:")
    print(f"  Input shape: {x.shape} → Output shape: {y.shape}")
    print(f"  Output range: [{y.min():.4f}, {y.max():.4f}]")
    print(f"  Mean: {y.mean():.4f}")
    return y


def visualize_activations():
    # defining input range for visualization
    x = torch.linspace(-5, 5, steps=100)
    # defining manual and inbuilt activations
    manual_activations = {
        "ReLU": ReLU_Manual(),
        "Sigmoid": Sigmoid_Manual(),
        "Tanh": Tanh_Manual(),
        "LeakyReLU": LeakyReLU_Manual(),
        "ELU": ELU_Manual(),
        "GELU": GELU_Manual(),
        "Softmax": Softmax_Manual(dim=0),
        "Softplus": Softplus_Manual(),
        "Swish": Swish_Manual(),
        "Mish": Mish_Manual()
    }
    inbuilt_activations = {
        "ReLU": ReLU_BuiltIn(),
        "Sigmoid": Sigmoid_BuiltIn(),
        "Tanh": Tanh_BuiltIn(),
        "Softmax": Softmax_BuiltIn(dim=0),
        "Softplus": Softplus_BuiltIn(),
        "Swish": Swish_BuiltIn(),
        "Mish": Mish_BuiltIn()
    }

    # creating subplots for manual and inbuilt activations
    fig1, axes1 = plt.subplots(3, 4, figsize=(10,8))
    fig2, axes2 = plt.subplots(3, 4, figsize=(10,8))

    #flattening axes for easier iteration
    axes1 = axes1.flatten()
    axes2 = axes2.flatten()

    # looping through manual activations
    for i, (name, activation) in enumerate(manual_activations.items()):
        y = activation(x)
        axes1[i].plot(x.numpy(), y.detach().numpy())
        axes1[i].set_title(f"Manual {name}")
        axes1[i].grid()

    # looping through inbuilt activations
    for i, (name, activation) in enumerate(inbuilt_activations.items()):
        y = activation(x)
        axes2[i].plot(x.numpy(), y.detach().numpy())
        axes2[i].set_title(f"Inbuilt {name}")
        axes2[i].grid()

    # show the plots
    plt.show()

    # save the figures
    fig1.savefig("manual_activations.png")
    fig2.savefig("inbuilt_activations.png")


def compare_derivatives():
    """Trace how loss flows backward through network layers

    Task: Network learns to output sum(input)
    Goal: Visualize vanishing/exploding gradients during backprop
    """
    """
    Thoughts and ideas:
    1) I need the names of the functions that i have and these can be used to call the specific activation functions. I already have separate funtions for manual and inbuild
    2) then i need to train the NN and in the process I need:
        a) input tensor of size 10
        b) a dictionary to store activations for each layer 
        c) a reference output and the corresponding loss i.e. the target function would be a sum of input tensor and the loss is the delta
        d) a dictionary to store gradients during backprop for each layer
            1) gradients require derivatives and i don't know how i'm gonna get that 
            2) do i also plot the loss landscape while i'm doing that?
    3) after this i need to plot the activations and gradients
        a) I decided to have one plot for each activation function 
        b) x - axis would be layer id and y axis is dual axis with the same scale for activations and gradients 
            1) while the overall loss is a scalar, gradients are neuron level during backprop so how do i visualize that? I have 10 connections in layer 1, 100 connections in layer 2, and 10 connections in layer 3.
        
    """
    activation_names = [
        'relu', 'sigmoid', 'tanh', 'leaky_relu',
        'elu', 'gelu', 'softmax', 'softplus',
        'swish', 'mish'
    ]

    results = {}

    for act_name in activation_names:
        # Create network with this activation
        net = SimpleNN(activation_type=act_name)

        # Create input: random values
        x = torch.randn(1, 10, requires_grad=True)
        target = x.sum()  # Target: sum of all input values

        # Storage for gradients at each layer
        activations = {}
        gradients = {}

        # Hook function to capture gradients during backward pass
        def make_hook(layer_id):
            def hook(grad):
                gradients[layer_id] = grad.clone()
            return hook

        # Forward pass: step through layers manually
        a = x

        # Layer 1 → Activation 1
        a = net.layer1(a)
        activations[0] = a.clone().detach()
        a = net.activation(a)
        a.register_hook(make_hook(0))

        # Layer 2 → Activation 2
        a = net.layer2(a)
        activations[1] = a.clone().detach()
        a = net.activation(a)
        a.register_hook(make_hook(1))

        # Layer 3 (no activation after final layer)
        a = net.layer3(a)
        activations[2] = a.clone().detach()
        a.register_hook(make_hook(2))

        # Compute loss: MSE between output and target sum
        loss = (a.squeeze() - target) ** 2

        # Backward pass: hooks capture gradients automatically
        loss.backward()

        # Compute gradient statistics for each layer
        gradient_stats = {}
        for layer_id in range(3):
            grad = gradients[layer_id]
            gradient_stats[layer_id] = {
                'sum': grad.abs().sum().item(),
                'mean': grad.abs().mean().item(),
                'max': grad.abs().max().item()
            }

        results[act_name] = gradient_stats

    # Create 2x5 subplot grid
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    axes = axes.flatten()

    layer_ids = [0, 1, 2]

    for idx, (name, stats) in enumerate(results.items()):
        ax = axes[idx]

        # Extract sum, mean, max for each layer
        sums = [stats[i]['sum'] for i in layer_ids]
        means = [stats[i]['mean'] for i in layer_ids]
        maxs = [stats[i]['max'] for i in layer_ids]

        # Plot all three on same subplot
        ax.plot(layer_ids, sums, 'o-', label='Sum', color='blue')
        ax.plot(layer_ids, means, 's-', label='Mean', color='red')
        ax.plot(layer_ids, maxs, '^-', label='Max', color='green')

        ax.set_xlabel('Layer')
        ax.set_ylabel('Gradient Magnitude')
        ax.set_title(f'{name}')
        ax.legend(fontsize=8)
        ax.grid()

    plt.tight_layout()
    plt.savefig("compare_derivatives.png")
    plt.show()



def get_activation_manual(name):
    """Utility to get activation by name"""
    activations = {
        "relu": ReLU_Manual(),
        "sigmoid": Sigmoid_Manual(),
        "tanh": Tanh_Manual(),
        "leaky_relu": LeakyReLU_Manual(),
        "elu": ELU_Manual(),
        "gelu": GELU_Manual(),
        "softmax": Softmax_Manual(dim=1),
        "softplus": Softplus_Manual(),
        "swish": Swish_Manual(),
        "mish": Mish_Manual()
    }
    return activations.get(name.lower(), None)

def get_activation_inbuilt(name):
    """Utility to get inbuilt activation by name"""
    activations = {
        "relu": ReLU_BuiltIn(),
        "sigmoid": Sigmoid_BuiltIn(),
        "tanh": Tanh_BuiltIn(),
        "softmax": Softmax_BuiltIn(dim=1),
        "softplus": Softplus_BuiltIn(),
        "swish": Swish_BuiltIn(),
        "mish": Mish_BuiltIn()
    }
    return activations.get(name.lower(), None)
# ============================================================================
# Simple Neural Network
# ============================================================================

class SimpleNN(nn.Module):
    """Neural network demonstrating activation usage

    TODO:
    1. __init__: Create linear layers and activation function
    2. forward: Apply layers with activations
       - Don't apply activation after final layer!
    """
    def __init__(self, input_dim = 10, hidden_dim=10, activation_type='relu', output_dim=1):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)
        self.activation = get_activation_manual(activation_type)
        

    def forward(self, x):
        # TODO: Implement
        # Pattern: Linear -> Activation -> Linear -> Activation -> Linear (no activation)
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.layer3(x)
        return x


# ============================================================================
# TESTING - RUN THIS TO CHECK YOUR IMPLEMENTATIONS
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Activation Functions Implementation")
    print("=" * 60)

    # Test 1: Basic activation function tests
    print("\n[1] Testing individual activations...\n")
    x = torch.randn(5, 10)

    test_activations = [
        ('ReLU', ReLU_Manual()),
        ('Sigmoid', Sigmoid_Manual()),
        ('Tanh', Tanh_Manual()),
        ('LeakyReLU', LeakyReLU_Manual()),
        ('ELU', ELU_Manual()),
        ('GELU', GELU_Manual()),
        ('Softmax', Softmax_Manual(dim=1)),
        ('Softplus', Softplus_Manual()),
    ]

    for name, activation in test_activations:
        y = test_activation(activation, x, name)

    # Test 2: SimpleNN with different activations
    print("\n[2] Testing SimpleNN with different activations...\n")
    x_test = torch.randn(1, 10)

    for act_type in ['relu', 'sigmoid', 'tanh']:
        net = SimpleNN(activation_type=act_type)
        output = net(x_test)
        print(f"SimpleNN with {act_type}: input shape {x_test.shape} → output shape {output.shape}")

    # Test 3: Visualize all activations
    print("\n[3] Generating activation visualizations...")
    print("   This will create two plots: manual_activations.png and inbuilt_activations.png")
    visualize_activations()

    # Test 4: Compare derivatives and gradient flow
    print("\n[4] Analyzing gradient flow through network layers...")
    print("   This will create: compare_derivatives.png")
    print("   (Shows vanishing/exploding gradient phenomenon)\n")
    compare_derivatives()

    print("=" * 60)
    print("✓ All tests completed!")
    print("=" * 60)

