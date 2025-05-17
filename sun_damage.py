import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import Slider

class SunDamageModel:
    def __init__(self, size=100, dt=1.0):
        """
        Initialize the sun damage pattern model.
        
        Parameters:
        - size: Size of the grid (size x size)
        - dt: Time step
        """
        self.size = size
        self.dt = dt
        
        # Initialize the grid
        self.U = np.ones((size, size))  # Activator (UV damage)
        self.V = np.zeros((size, size))  # Inhibitor (melanin protection)
        self.D = np.zeros((size, size))  # DNA damage
        
        # Set parameters for sun damage
        # Parameters tuned to match UV damage dynamics
        self.Du, self.Dv = 0.16, 0.08  # Diffusion rates
        self.f, self.k = 0.0545, 0.062  # Feed and kill rates
        self.uv_intensity = 0.02       # UV radiation intensity
        self.melanin_rate = 0.01       # Rate of melanin production
        self.dna_repair_rate = 0.005   # Rate of DNA repair
        
        # Initialize sun damage pattern
        self.init_sun_damage()
            
    def init_sun_damage(self):
        """Initialize sun damage pattern with biologically realistic conditions."""
        # Create initial conditions based on skin structure
        # Start with a uniform skin surface
        x = np.linspace(-1, 1, self.size)
        y = np.linspace(-1, 1, self.size)
        X, Y = np.meshgrid(x, y)
        
        # Create initial UV exposure pattern
        # This represents the initial UV radiation
        uv_pattern = np.sin(8 * X) * np.cos(8 * Y)  # Simulating UV light patterns
        self.V = 0.5 + 0.5 * uv_pattern
        
        # Initialize melanin distribution
        # Higher in areas with more UV exposure
        self.U = 1 - 0.5 * np.abs(uv_pattern)
        
        # Initialize DNA damage
        # Higher in areas with more UV exposure
        self.D = 0.2 * np.abs(uv_pattern)
        
        # Add biological noise to simulate skin variability
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
        Ld = self.laplacian(self.D)
        
        # Calculate the reaction terms
        # This represents the interaction between UV damage and melanin
        uv2 = self.U[1:-1, 1:-1] * self.V[1:-1, 1:-1] * self.V[1:-1, 1:-1]
        
        # Update DNA damage (diffusion and UV effects)
        self.D[1:-1, 1:-1] += self.dt * (0.1 * Ld + self.uv_intensity * self.V[1:-1, 1:-1] - self.dna_repair_rate)
        self.D = np.clip(self.D, 0, 1)
        
        # Melanin production effect based on UV exposure
        melanin_effect = 1 + self.melanin_rate * self.V[1:-1, 1:-1]
        
        # Update the grid
        # Add a small amount of noise to simulate biological variability
        noise = np.random.normal(0, 0.001, (self.size-2, self.size-2))
        
        self.U[1:-1, 1:-1] += self.dt * (self.Du * Lu - uv2 + self.f * (1 - self.U[1:-1, 1:-1])) + noise
        self.V[1:-1, 1:-1] += self.dt * (self.Dv * Lv + uv2 * melanin_effect - (self.f + self.k) * self.V[1:-1, 1:-1]) + noise
        
        # Enforce boundary conditions
        self.U[0, :] = self.U[1, :]
        self.U[-1, :] = self.U[-2, :]
        self.U[:, 0] = self.U[:, 1]
        self.U[:, -1] = self.U[:, -2]
        
        self.V[0, :] = self.V[1, :]
        self.V[-1, :] = self.V[-2, :]
        self.V[:, 0] = self.V[:, 1]
        self.V[:, -1] = self.V[:, -2]
        
        self.D[0, :] = 0  # No DNA damage at boundaries
        self.D[-1, :] = 0
        self.D[:, 0] = 0
        self.D[:, -1] = 0

def create_custom_colormap():
    """Create a custom colormap with the specified colors."""
    colors = ['#FFA0AC', '#B4DC7F']  # Pink to Green
    return LinearSegmentedColormap.from_list('custom', colors)

def main():
    # Create the model
    model = SunDamageModel(size=100)
    
    # Create figure with white background
    fig = plt.figure(figsize=(10, 8), facecolor='white')
    ax = fig.add_subplot(111)
    
    # Create custom colormap
    custom_cmap = create_custom_colormap()
    
    # Set up the plot
    img = ax.imshow(model.V, cmap=custom_cmap, interpolation='bilinear')
    plt.colorbar(img, ax=ax)
    ax.set_title('Sun Damage Dynamics\n(Simulating UV Effects & Melanin Response)')
    
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