import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import Slider

class CellMembraneModel:
    def __init__(self, size=100, dt=1.0):
        """
        Initialize the cell membrane pattern model.
        
        Parameters:
        - size: Size of the grid (size x size)
        - dt: Time step
        """
        self.size = size
        self.dt = dt
        
        # Initialize the grid
        self.U = np.ones((size, size))  # Activator (lipid domain activator)
        self.V = np.zeros((size, size))  # Inhibitor (lipid domain inhibitor)
        self.T = np.ones((size, size))   # Membrane tension
        
        # Set parameters for cell membrane
        # Parameters tuned to match lipid domain formation
        self.Du, self.Dv = 0.14, 0.06  # Diffusion rates
        self.f, self.k = 0.035, 0.065  # Feed and kill rates
        self.tension_rate = 0.01       # Rate of tension propagation
        self.domain_threshold = 0.3    # Threshold for domain formation
        
        # Initialize cell membrane
        self.init_membrane()
            
    def init_membrane(self):
        """Initialize cell membrane with biologically realistic conditions."""
        # Create initial conditions based on membrane structure
        # Start with a circular membrane
        x = np.linspace(-1, 1, self.size)
        y = np.linspace(-1, 1, self.size)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        
        # Create initial membrane structure
        # This represents the initial lipid distribution
        membrane = np.exp(-5 * R)  # Circular membrane
        
        # Add initial lipid domains
        # These represent the initial lipid rafts
        domains = np.random.random((self.size, self.size)) > 0.95
        self.V = membrane * (0.5 + 0.5 * domains)
        
        # Initialize membrane tension
        # Higher at edges, lower in center (simulating membrane curvature)
        self.T = 1 - 0.5 * membrane
        
        # Add biological noise to simulate molecular fluctuations
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
        Lt = self.laplacian(self.T)
        
        # Calculate the reaction terms
        # This represents the interaction between lipid domains
        uv2 = self.U[1:-1, 1:-1] * self.V[1:-1, 1:-1] * self.V[1:-1, 1:-1]
        
        # Update membrane tension (diffusion and domain effects)
        self.T[1:-1, 1:-1] += self.dt * (self.tension_rate * Lt - 0.1 * self.V[1:-1, 1:-1])
        self.T = np.clip(self.T, 0, 1)
        
        # Domain formation effect based on tension (only for inner grid)
        domain_effect = np.where(self.T[1:-1, 1:-1] > self.domain_threshold, 1.2, 1.0)
        
        # Update the grid
        # Add a small amount of noise to simulate molecular fluctuations
        noise = np.random.normal(0, 0.001, (self.size-2, self.size-2))
        
        self.U[1:-1, 1:-1] += self.dt * (self.Du * Lu - uv2 + self.f * (1 - self.U[1:-1, 1:-1])) + noise
        self.V[1:-1, 1:-1] += self.dt * (self.Dv * Lv + uv2 * domain_effect - (self.f + self.k) * self.V[1:-1, 1:-1]) + noise
        
        # Enforce boundary conditions
        self.U[0, :] = self.U[1, :]
        self.U[-1, :] = self.U[-2, :]
        self.U[:, 0] = self.U[:, 1]
        self.U[:, -1] = self.U[:, -2]
        
        self.V[0, :] = self.V[1, :]
        self.V[-1, :] = self.V[-2, :]
        self.V[:, 0] = self.V[:, 1]
        self.V[:, -1] = self.V[:, -2]
        
        self.T[0, :] = 1  # Constant tension at boundaries
        self.T[-1, :] = 1
        self.T[:, 0] = 1
        self.T[:, -1] = 1

def create_custom_colormap():
    """Create a custom colormap with the specified colors."""
    colors = ['#EFCFE3', '#B3DEE2']  # Soft pink to light blue
    return LinearSegmentedColormap.from_list('custom', colors)

def main():
    # Create the model
    model = CellMembraneModel(size=100)
    
    # Create figure with white background
    fig = plt.figure(figsize=(10, 8), facecolor='white')
    ax = fig.add_subplot(111)
    
    # Create custom colormap
    custom_cmap = create_custom_colormap()
    
    # Set up the plot
    img = ax.imshow(model.V, cmap=custom_cmap, interpolation='bilinear')
    plt.colorbar(img, ax=ax)
    ax.set_title('Cell Membrane Dynamics\n(Simulating Lipid Domain Formation)')
    
    # Add speed control slider
    ax_slider = plt.axes([0.2, 0.02, 0.6, 0.03])
    speed_slider = Slider(ax_slider, 'Speed', 1, 100, valinit=50)
    
    def update(frame):
        model.update()
        img.set_array(model.V)
        return [img]
    
    # Create the animation with proper parameters
    anim = FuncAnimation(fig, update, frames=200, interval=50, blit=True, cache_frame_data=False)
    
    def update_speed(val):
        anim.event_source.interval = 1000 / val
    
    speed_slider.on_changed(update_speed)
    plt.show()

if __name__ == "__main__":
    main() 