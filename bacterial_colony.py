import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import Slider

class BacterialColonyModel:
    def __init__(self, size=100, dt=1.0):
        """
        Initialize the bacterial colony pattern model.
        
        Parameters:
        - size: Size of the grid (size x size)
        - dt: Time step
        """
        self.size = size
        self.dt = dt
        
        # Initialize the grid
        self.U = np.ones((size, size))  # Activator (bacterial density)
        self.V = np.zeros((size, size))  # Inhibitor (nutrient concentration)
        self.N = np.ones((size, size))   # Nutrients
        
        # Set parameters for bacterial colony
        # Parameters tuned to match bacterial growth dynamics
        self.Du, self.Dv = 0.16, 0.08  # Diffusion rates
        self.f, self.k = 0.0545, 0.062  # Feed and kill rates
        self.nutrient_consumption = 0.01  # Rate of nutrient consumption
        self.quorum_threshold = 0.5      # Threshold for quorum sensing
        
        # Initialize bacterial colony
        self.init_colony()
            
    def init_colony(self):
        """Initialize bacterial colony with biologically realistic conditions."""
        # Create initial conditions based on bacterial inoculation
        # Start with a central inoculation point
        x = np.linspace(-1, 1, self.size)
        y = np.linspace(-1, 1, self.size)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        
        # Create initial bacterial colony
        # This represents the initial bacterial inoculation
        colony = np.exp(-10 * R)  # Central inoculation point
        
        # Add initial bacterial density
        self.V = colony
        
        # Initialize nutrient concentration
        # Higher at edges, lower in center (simulating nutrient diffusion)
        self.N = 1 - 0.5 * colony
        
        # Add biological noise to simulate cell-to-cell variability
        noise = np.random.normal(0, 0.02, (self.size, self.size))
        self.V += noise
        self.V = np.clip(self.V, 0, 1)
        
        # Initialize activator concentration
        self.U = 1 - self.V  # Inverse relationship between activator and inhibitor

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
        Ln = self.laplacian(self.N)
        
        # Calculate the reaction terms
        # This represents the interaction between bacterial density and nutrients
        uv2 = self.U[1:-1, 1:-1] * self.V[1:-1, 1:-1] * self.V[1:-1, 1:-1]
        
        # Update nutrients (diffusion and consumption)
        self.N[1:-1, 1:-1] += self.dt * (0.1 * Ln - self.nutrient_consumption * self.V[1:-1, 1:-1])
        self.N = np.clip(self.N, 0, 1)
        
        # Quorum sensing effect
        quorum_effect = np.where(self.V > self.quorum_threshold, 1.2, 1.0)
        
        # Update the grid
        # Add a small amount of noise to simulate biological variability
        noise = np.random.normal(0, 0.001, (self.size-2, self.size-2))
        
        self.U[1:-1, 1:-1] += self.dt * (self.Du * Lu - uv2 + self.f * (1 - self.U[1:-1, 1:-1])) + noise
        self.V[1:-1, 1:-1] += self.dt * (self.Dv * Lv + uv2 * quorum_effect - (self.f + self.k) * self.V[1:-1, 1:-1]) + noise
        
        # Enforce boundary conditions
        self.U[0, :] = self.U[1, :]
        self.U[-1, :] = self.U[-2, :]
        self.U[:, 0] = self.U[:, 1]
        self.U[:, -1] = self.U[:, -2]
        
        self.V[0, :] = self.V[1, :]
        self.V[-1, :] = self.V[-2, :]
        self.V[:, 0] = self.V[:, 1]
        self.V[:, -1] = self.V[:, -2]
        
        self.N[0, :] = 1  # Constant nutrient supply at boundaries
        self.N[-1, :] = 1
        self.N[:, 0] = 1
        self.N[:, -1] = 1

def create_custom_colormap():
    """Create a custom colormap with the specified colors."""
    colors = ['#FFA0AC', '#B4DC7F']  # Pink to Green
    return LinearSegmentedColormap.from_list('custom', colors)

def main():
    # Create the model
    model = BacterialColonyModel(size=100)
    
    # Create figure with white background
    fig = plt.figure(figsize=(10, 8), facecolor='white')
    ax = fig.add_subplot(111)
    
    # Create custom colormap
    custom_cmap = create_custom_colormap()
    
    # Set up the plot
    img = ax.imshow(model.V, cmap=custom_cmap, interpolation='bilinear')
    plt.colorbar(img, ax=ax)
    ax.set_title('Bacterial Colony Formation\n(Simulating Growth Dynamics & Quorum Sensing)')
    
    # Add speed control slider
    ax_slider = plt.axes([0.2, 0.02, 0.6, 0.03])
    speed_slider = Slider(ax_slider, 'Speed', 1, 100, valinit=50)
    
    def update(frame):
        model.update()
        img.set_array(model.V)
        return [img]
    
    # Create the animation with variable speed
    anim = FuncAnimation(fig, update, frames=200, interval=50, blit=True)
    
    def update_speed(val):
        anim.event_source.interval = 1000 / val
    
    speed_slider.on_changed(update_speed)
    plt.show()

if __name__ == "__main__":
    main() 