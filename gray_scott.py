import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class GrayScottModel:
    def __init__(self, size=100, dt=1.0, Du=0.16, Dv=0.08, f=0.035, k=0.065):
        """
        Initialize the Gray-Scott model.
        
        Parameters:
        - size: Size of the grid (size x size)
        - dt: Time step
        - Du: Diffusion rate of chemical U
        - Dv: Diffusion rate of chemical V
        - f: Feed rate
        - k: Kill rate
        """
        self.size = size
        self.dt = dt
        self.Du = Du
        self.Dv = Dv
        self.f = f
        self.k = k
        
        # Initialize the grid with random values
        self.U = np.ones((size, size))
        self.V = np.zeros((size, size))
        
        # Add some random noise to create initial patterns
        r = 20
        self.U[size//2-r:size//2+r, size//2-r:size//2+r] = 0.5
        self.V[size//2-r:size//2+r, size//2-r:size//2+r] = 0.25
        self.U += np.random.random((size, size)) * 0.1
        self.V += np.random.random((size, size)) * 0.1

    def laplacian(self, Z):
        """Calculate the Laplacian of the grid using a 3x3 convolution."""
        Ztop = Z[0:-2, 1:-1]
        Zleft = Z[1:-1, 0:-2]
        Zbottom = Z[2:, 1:-1]
        Zright = Z[1:-1, 2:]
        Zcenter = Z[1:-1, 1:-1]
        return (Ztop + Zleft + Zbottom + Zright - 4 * Zcenter)

    def update(self):
        """Update the grid for one time step."""
        # Calculate the Laplacian
        Lu = self.laplacian(self.U)
        Lv = self.laplacian(self.V)
        
        # Calculate the reaction terms
        uv2 = self.U[1:-1, 1:-1] * self.V[1:-1, 1:-1] * self.V[1:-1, 1:-1]
        
        # Update the grid
        self.U[1:-1, 1:-1] += self.dt * (self.Du * Lu - uv2 + self.f * (1 - self.U[1:-1, 1:-1]))
        self.V[1:-1, 1:-1] += self.dt * (self.Dv * Lv + uv2 - (self.f + self.k) * self.V[1:-1, 1:-1])
        
        # Enforce boundary conditions
        self.U[0, :] = self.U[1, :]
        self.U[-1, :] = self.U[-2, :]
        self.U[:, 0] = self.U[:, 1]
        self.U[:, -1] = self.U[:, -2]
        
        self.V[0, :] = self.V[1, :]
        self.V[-1, :] = self.V[-2, :]
        self.V[:, 0] = self.V[:, 1]
        self.V[:, -1] = self.V[:, -2]

def main():
    # Create the model
    model = GrayScottModel(size=100)
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(8, 8))
    img = ax.imshow(model.V, cmap='viridis', interpolation='bilinear')
    plt.colorbar(img, ax=ax)
    
    def update(frame):
        model.update()
        img.set_array(model.V)
        return [img]
    
    # Create the animation
    anim = FuncAnimation(fig, update, frames=200, interval=50, blit=True)
    plt.show()

if __name__ == "__main__":
    main() 