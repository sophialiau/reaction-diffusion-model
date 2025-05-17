import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import Slider

class SkinWrinklingModel:
    def __init__(self, size=100, dt=1.0):
        """
        Initialize the skin wrinkling pattern model.
        
        Parameters:
        - size: Size of the grid (size x size)
        - dt: Time step
        """
        self.size = size
        self.dt = dt
        
        # Initialize the grid
        self.U = np.ones((size, size))  # Activator (mechanical stress)
        self.V = np.zeros((size, size))  # Inhibitor (collagen resistance)
        self.C = np.zeros((size, size))  # Collagen orientation
        
        # Set parameters for skin wrinkling
        # Parameters tuned to match skin mechanics
        self.Du, self.Dv = 0.14, 0.06  # Diffusion rates
        self.f, self.k = 0.035, 0.065  # Feed and kill rates
        self.stress_rate = 0.02        # Rate of stress accumulation
        self.collagen_rate = 0.01      # Rate of collagen reorganization
        
        # Initialize skin wrinkling pattern
        self.init_wrinkling()
            
    def init_wrinkling(self):
        """Initialize skin wrinkling pattern with biologically realistic conditions."""
        # Create initial conditions based on skin structure
        # Start with a uniform skin surface
        x = np.linspace(-1, 1, self.size)
        y = np.linspace(-1, 1, self.size)
        X, Y = np.meshgrid(x, y)
        
        # Create initial collagen orientation with more natural variation
        # This represents the initial collagen fiber network
        theta = np.arctan2(Y, X)
        self.C = np.sin(4 * theta) + 0.5 * np.sin(2 * theta) + 0.2 * np.random.random((self.size, self.size))
        
        # Add initial mechanical stress with more natural variation
        # This represents the initial skin tension
        stress = np.random.normal(0, 0.1, (self.size, self.size))
        self.V = 0.5 + stress + 0.1 * np.sin(4 * X) * np.cos(4 * Y)
        
        # Initialize collagen resistance with more natural variation
        # Higher in areas with aligned collagen
        self.U = 1 - 0.5 * np.abs(self.C) + 0.1 * np.random.random((self.size, self.size))
        
        # Add biological noise to simulate tissue variability
        noise = np.random.normal(0, 0.02, (self.size, self.size))
        self.V += noise
        self.V = np.clip(self.V, 0, 1)

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
        Lc = self.laplacian(self.C)
        
        # Calculate the reaction terms with more organic variation
        # This represents the interaction between mechanical stress and collagen
        uv2 = self.U[1:-1, 1:-1] * self.V[1:-1, 1:-1] * self.V[1:-1, 1:-1]
        
        # Add temporal variation to parameters
        stress_var = self.stress_rate * (1 + 0.1 * np.sin(0.1 * np.random.random()))
        collagen_var = self.collagen_rate * (1 + 0.1 * np.cos(0.1 * np.random.random()))
        
        # Update collagen orientation with more organic dynamics
        self.C[1:-1, 1:-1] += self.dt * (0.1 * Lc + collagen_var * self.V[1:-1, 1:-1] * 
                                        np.sin(2 * np.pi * self.C[1:-1, 1:-1]) * 
                                        (1 + 0.2 * np.sin(0.05 * self.C[1:-1, 1:-1])))
        self.C = np.clip(self.C, -1, 1)
        
        # Stress accumulation effect with more natural variation
        stress_effect = 1 + stress_var * (np.abs(self.C[1:-1, 1:-1]) + 0.5 * self.V[1:-1, 1:-1]) * \
                       (1 + 0.1 * np.sin(0.2 * self.V[1:-1, 1:-1]))
        
        # Update the grid with more organic dynamics
        # Add a small amount of noise to simulate tissue variability
        noise = np.random.normal(0, 0.001, (self.size-2, self.size-2))
        
        # Update with feedback from collagen orientation and natural variation
        self.U[1:-1, 1:-1] += self.dt * (self.Du * Lu - uv2 + self.f * (1 - self.U[1:-1, 1:-1]) + 
                                        0.1 * np.abs(self.C[1:-1, 1:-1]) * (1 + 0.1 * np.sin(0.1 * self.U[1:-1, 1:-1]))) + noise
        self.V[1:-1, 1:-1] += self.dt * (self.Dv * Lv + uv2 * stress_effect - 
                                        (self.f + self.k) * self.V[1:-1, 1:-1]) + noise
        
        # Enforce boundary conditions with smooth transitions
        self.U[0, :] = self.U[1, :]
        self.U[-1, :] = self.U[-2, :]
        self.U[:, 0] = self.U[:, 1]
        self.U[:, -1] = self.U[:, -2]
        
        self.V[0, :] = self.V[1, :]
        self.V[-1, :] = self.V[-2, :]
        self.V[:, 0] = self.V[:, 1]
        self.V[:, -1] = self.V[:, -2]
        
        self.C[0, :] = 0  # No collagen orientation at boundaries
        self.C[-1, :] = 0
        self.C[:, 0] = 0
        self.C[:, -1] = 0

def create_custom_colormap():
    """Create a custom colormap with the specified colors."""
    colors = ['#EFCFE3', '#B3DEE2']  # Soft pink to light blue
    return LinearSegmentedColormap.from_list('custom', colors)

def main():
    # Create the model
    model = SkinWrinklingModel(size=100)
    
    # Create figure with white background
    fig = plt.figure(figsize=(10, 8), facecolor='white')
    ax = fig.add_subplot(111)
    
    # Create custom colormap
    custom_cmap = create_custom_colormap()
    
    # Set up the plot
    img = ax.imshow(model.V, cmap=custom_cmap, interpolation='bilinear')
    plt.colorbar(img, ax=ax)
    ax.set_title('Skin Wrinkling Dynamics\n(Simulating Mechanical Stress & Collagen Reorganization)')
    
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