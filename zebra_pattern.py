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
        
        # Add multiple Turing instability seed points with more organic distribution
        # These represent the initial melanocyte activation centers
        num_seeds = 20  # Increased number of seed points
        seed_points = np.zeros((self.size, self.size))
        
        # Create seed points in a more natural pattern
        for _ in range(num_seeds):
            # Random position with bias towards the center
            x = int(np.random.normal(self.size/2, self.size/6))
            y = int(np.random.normal(self.size/2, self.size/6))
            x = np.clip(x, 0, self.size-1)
            y = np.clip(y, 0, self.size-1)
            
            # Add a small cluster of activation
            radius = np.random.randint(2, 5)
            for i in range(-radius, radius+1):
                for j in range(-radius, radius+1):
                    if 0 <= x+i < self.size and 0 <= y+j < self.size:
                        distance = np.sqrt(i*i + j*j)
                        if distance <= radius:
                            intensity = 0.5 * (1 - distance/radius)
                            seed_points[x+i, y+j] = max(seed_points[x+i, y+j], intensity)
        
        self.V = seed_points
        
        # Add developmental timing gradient with more natural variation
        # This simulates the wave of pattern formation from back to front
        timing = np.exp(-5 * (1 - gradient)) * (1 + 0.2 * np.sin(8 * gradient))
        self.V = self.V * timing
        
        # Add more organic noise to simulate biological variability
        noise = np.random.normal(0, 0.05, (self.size, self.size))
        self.V += noise
        self.V = np.clip(self.V, 0, 1)
        
        # Initialize activator concentration with more natural variation
        self.U = 1 - self.V + 0.1 * np.random.random((self.size, self.size))

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
        
        # Calculate the reaction terms with more organic variation
        # This represents the interaction between melanocyte activators and inhibitors
        uv2 = self.U[1:-1, 1:-1] * self.V[1:-1, 1:-1] * self.V[1:-1, 1:-1]
        
        # Add temporal variation to feed and kill rates
        f_var = self.f * (1 + 0.1 * np.sin(0.1 * np.random.random()))
        k_var = self.k * (1 + 0.1 * np.cos(0.1 * np.random.random()))
        
        # Update the grid with more organic dynamics
        # Add a small amount of noise to simulate biological variability
        noise = np.random.normal(0, 0.001, (self.size-2, self.size-2))
        
        # Add more dynamic stripe formation
        stripe_effect = 1 + 0.2 * np.sin(0.1 * self.V[1:-1, 1:-1])
        
        self.U[1:-1, 1:-1] += self.dt * (self.Du * Lu - uv2 + f_var * (1 - self.U[1:-1, 1:-1])) + noise
        self.V[1:-1, 1:-1] += self.dt * (self.Dv * Lv + uv2 * stripe_effect - 
                                        (f_var + k_var) * self.V[1:-1, 1:-1]) + noise
        
        # Enforce boundary conditions with smooth transitions
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
    colors = ['#EFCFE3', '#B3DEE2']  # Soft pink to light blue
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
    
    # Create the animation with proper parameters
    anim = FuncAnimation(fig, update, frames=200, interval=50, blit=True, cache_frame_data=False)
    
    def update_speed(val):
        anim.event_source.interval = 1000 / val
    
    speed_slider.on_changed(update_speed)
    plt.show()

if __name__ == "__main__":
    main() 