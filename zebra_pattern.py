import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import Slider

class ZebraPatternModel:
    def __init__(self, size=100, dt=1.0):
        """
        Initialize the zebra stripe pattern model.
        
        Parameters:
        - size: Size of the grid (size x size)
        - dt: Time step
        """
        self.size = size
        self.dt = dt
        
        # Initialize the grid
        self.U = np.ones((size, size))  # Activator (melanocyte activator)
        self.V = np.zeros((size, size))  # Inhibitor (melanocyte inhibitor)
        
        # Set parameters for zebra stripes
        # Parameters tuned to match melanocyte behavior and stripe formation
        self.Du, self.Dv = 0.16, 0.08  # Diffusion rates
        self.f, self.k = 0.035, 0.065  # Feed and kill rates
        
        # Initialize zebra pattern
        self.init_zebra()
            
    def init_zebra(self):
        """Initialize zebra stripe pattern with biologically realistic conditions."""
        # Create initial conditions based on embryonic development
        # Start with a gradient from back to front (dorsal-ventral axis)
        y = np.linspace(0, 1, self.size)
        gradient = np.tile(y, (self.size, 1)).T
        
        # Add Turing instability seed points
        # These represent the initial melanocyte activation centers
        seed_points = np.random.random((self.size, self.size)) > 0.98
        self.V[seed_points] = 0.5
        
        # Add developmental timing gradient
        # This simulates the wave of pattern formation from back to front
        timing = np.exp(-5 * (1 - gradient))
        self.V = self.V * timing
        
        # Add some noise to simulate biological variability
        noise = np.random.normal(0, 0.05, (self.size, self.size))
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
        
        # Calculate the reaction terms
        # This represents the interaction between melanocyte activators and inhibitors
        uv2 = self.U[1:-1, 1:-1] * self.V[1:-1, 1:-1] * self.V[1:-1, 1:-1]
        
        # Update the grid
        # Add a small amount of noise to simulate biological variability
        noise = np.random.normal(0, 0.001, (self.size-2, self.size-2))
        
        self.U[1:-1, 1:-1] += self.dt * (self.Du * Lu - uv2 + self.f * (1 - self.U[1:-1, 1:-1])) + noise
        self.V[1:-1, 1:-1] += self.dt * (self.Dv * Lv + uv2 - (self.f + self.k) * self.V[1:-1, 1:-1]) + noise
        
        # Enforce boundary conditions
        self.U[0, :] = self.U[1, :]
        self.U[-1, :] = self.U[-2, :]
        self.U[:, 0] = self.U[:, 1]
        self.U[:, -1] = self.U[:, -2]
        
        self.V[0, :] = self.V[1, :]
        self.V[-1, :] = self.V[-2, :]
        self.V[:, 0] = self.V[:, 1]
        self.V[:, -1] = self.V[:, -2]

def create_custom_colormap():
    """Create a custom colormap with the specified colors."""
    colors = ['#FFA0AC', '#B4DC7F']  # Pink to Green
    return LinearSegmentedColormap.from_list('custom', colors)

def main():
    # Create the model
    model = ZebraPatternModel(size=100)
    
    # Create figure with white background
    fig = plt.figure(figsize=(10, 8), facecolor='white')
    ax = fig.add_subplot(111)
    
    # Create custom colormap
    custom_cmap = create_custom_colormap()
    
    # Set up the plot
    img = ax.imshow(model.V, cmap=custom_cmap, interpolation='bilinear')
    plt.colorbar(img, ax=ax)
    ax.set_title('Zebra Stripe Pattern Formation\n(Simulating Melanocyte Development)')
    
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