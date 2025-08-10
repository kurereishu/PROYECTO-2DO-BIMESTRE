import tkinter as tk
from tkinter import ttk
from tkcalendar import DateEntry
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from datetime import datetime, timedelta
from pytz import timezone
from pysolar.solar import get_altitude, get_azimuth
import math

# Configuración inicial (EPN - Quito)
LATITUDE = -0.2105367  
LONGITUDE = -78.491614
TIMEZONE = timezone("America/Guayaquil")
PANEL_WIDTH = 0.5
PANEL_HEIGHT = 0.8

class SolarTrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Seguidor Solar 2DOF - Simulación Realista")
        self.root.geometry("1200x800")
        
        # Inicialización de atributos
        self.animation = None
        self.sun_vectors = None
        self.times = None
        self.elevations = None
        self.azimuths = None
        self.current_frame = 0
        self.reference_vector = np.array([0, 1, 0])
        
        # Componentes gráficos
        self.fig = None
        self.ax = None
        self.canvas = None
        self.sun_path = None
        self.panel = None
        self.quiver = None
        self.sun_dot = None
        
        # Configuración de la interfaz
        self.setup_ui()
        
    def setup_ui(self):
        """Configura todos los elementos de la interfaz"""
        # Frame de controles
        control_frame = ttk.LabelFrame(self.root, text="Configuración", padding=10)
        control_frame.grid(row=0, column=0, padx=10, pady=10, sticky="n")

        # Controles de fecha/hora
        ttk.Label(control_frame, text="Fecha:").grid(row=0, column=0, sticky="w", pady=5)
        self.date_entry = DateEntry(control_frame, date_pattern='yyyy-mm-dd')
        self.date_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(control_frame, text="Hora inicio:").grid(row=1, column=0, sticky="w", pady=5)
        self.hour_spin = ttk.Spinbox(control_frame, from_=0, to=23, width=8)
        self.hour_spin.set(6)
        self.hour_spin.grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(control_frame, text="Duración (h):").grid(row=2, column=0, sticky="w", pady=5)
        self.duration_spin = ttk.Spinbox(control_frame, from_=1, to=12, width=8)
        self.duration_spin.set(12)
        self.duration_spin.grid(row=2, column=1, padx=5, pady=5)

        ttk.Label(control_frame, text="Intervalo (min):").grid(row=3, column=0, sticky="w", pady=5)
        self.interval_spin = ttk.Spinbox(control_frame, from_=1, to=60, width=8)
        self.interval_spin.set(15)
        self.interval_spin.grid(row=3, column=1, padx=5, pady=5)

        # Botón de control
        self.run_button = ttk.Button(control_frame, text="Simular Trayectoria", command=self.run_simulation)
        self.run_button.grid(row=4, column=0, columnspan=2, pady=15)

        # Panel de ángulos calculados
        angles_frame = ttk.LabelFrame(control_frame, text="Ángulos Calculados", padding=10)
        angles_frame.grid(row=5, column=0, columnspan=2, pady=10, sticky="we")

        ttk.Label(angles_frame, text="Elevación:").grid(row=0, column=0, sticky="w")
        self.elevation_label = ttk.Label(angles_frame, text="0.00°")
        self.elevation_label.grid(row=0, column=1, sticky="e")

        ttk.Label(angles_frame, text="Azimuth:").grid(row=1, column=0, sticky="w")
        self.azimuth_label = ttk.Label(angles_frame, text="0.00°")
        self.azimuth_label.grid(row=1, column=1, sticky="e")

        ttk.Label(angles_frame, text="Pitch:").grid(row=2, column=0, sticky="w")
        self.pitch_label = ttk.Label(angles_frame, text="0.00°")
        self.pitch_label.grid(row=2, column=1, sticky="e")

        ttk.Label(angles_frame, text="Roll:").grid(row=3, column=0, sticky="w")
        self.roll_label = ttk.Label(angles_frame, text="0.00°")
        self.roll_label.grid(row=3, column=1, sticky="e")

        # Frame de visualización 3D
        self.plot_frame = ttk.LabelFrame(self.root, text="Trayectoria Solar", padding=10)
        self.plot_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

    def calculate_sun_positions(self, start_date, duration_hours, time_step_minutes):
        """Calcula la trayectoria solar para un período"""
        times = []
        sun_vectors = []
        elevations = []
        azimuths = []
        
        for i in range(0, duration_hours * 60, time_step_minutes):
            current_time = start_date + timedelta(minutes=i)
            elevation = get_altitude(LATITUDE, LONGITUDE, current_time)
            azimuth = get_azimuth(LATITUDE, LONGITUDE, current_time)
            
            # Convertir a coordenadas cartesianas
            elev_rad = math.radians(elevation)
            azim_rad = math.radians(azimuth)
            
            x = math.cos(elev_rad) * math.sin(azim_rad)
            y = math.cos(elev_rad) * math.cos(azim_rad)
            z = math.sin(elev_rad)
            
            times.append(current_time)
            sun_vectors.append([x, y, z])
            elevations.append(elevation)
            azimuths.append(azimuth)
        
        return times, np.array(sun_vectors), elevations, azimuths

    def create_panel_vertices(self, normal_vector):
        """Crea vértices del panel orientado perpendicular al sol"""
        normal = normal_vector / np.linalg.norm(normal_vector)
        
        # Vector horizontal de referencia (Este)
        if np.allclose(np.abs(normal), [0, 0, 1]):
            u = np.array([1, 0, 0])  # Caso especial (sol cenital)
        else:
            u = np.cross(self.reference_vector, normal)
            u = u / np.linalg.norm(u)
        
        v = np.cross(normal, u)
        v = v / np.linalg.norm(v)
        
        half_w = PANEL_WIDTH / 2
        half_h = PANEL_HEIGHT / 2
        
        return [[
            half_h * u + half_w * v,
            -half_h * u + half_w * v,
            -half_h * u - half_w * v,
            half_h * u - half_w * v
        ]]

    def init_animation(self):
        """Prepara la visualización 3D inicial"""
        self.ax.clear()
        
        # Configuración de ejes
        self.ax.set_xlim([-1.5, 1.5])
        self.ax.set_ylim([-1.5, 1.5])
        self.ax.set_zlim([0, 1.5])
        self.ax.set_xlabel("Este-Oeste")
        self.ax.set_ylabel("Norte-Sur")
        self.ax.set_zlabel("Altura")
        
        # Trayectoria solar
        self.sun_path, = self.ax.plot(
            self.sun_vectors[:,0], 
            self.sun_vectors[:,1], 
            self.sun_vectors[:,2], 
            'y-', alpha=0.6, label="Trayectoria solar"
        )
        
        # Panel solar inicial
        panel_verts = self.create_panel_vertices(self.sun_vectors[0])
        self.panel = Poly3DCollection(panel_verts, color='blue', alpha=0.7, label="Panel solar")
        self.ax.add_collection3d(self.panel)
        
        # Texto de puntos cardinales (corregido para 3D)
        self.ax.text(1.4, 0, 0, 'Este', color='red')
        self.ax.text(-1.4, 0, 0, 'Oeste', color='red')
        self.ax.text(0, 1.4, 0, 'Norte', color='red')
        self.ax.text(0, -1.4, 0, 'Sur', color='red')
        
        return self.panel,

    def update_animation(self, frame):
        """Actualiza el frame de la animación"""
        self.current_frame = frame
        current_time = self.times[frame]
        
        # Actualizar panel solar
        normal = self.sun_vectors[frame]
        panel_verts = self.create_panel_vertices(normal)
        self.panel.set_verts(panel_verts)
        
        # Actualizar rayo solar
        if hasattr(self, 'quiver') and self.quiver is not None:
            self.quiver.remove()
        self.quiver = self.ax.quiver(
            0, 0, 0, normal[0], normal[1], normal[2], 
            color='orange', length=1.5, arrow_length_ratio=0.1
        )
        
        # Actualizar posición del sol
        if hasattr(self, 'sun_dot') and self.sun_dot is not None:
            self.sun_dot.remove()
        self.sun_dot = self.ax.scatter(
            [normal[0]], [normal[1]], [normal[2]],
            color='yellow', s=200, edgecolor='orange', zorder=10
        )
        
        # Actualizar ángulos en la interfaz
        time_str = current_time.strftime("%H:%M")
        self.ax.set_title(f"Posición solar: {time_str}")
        self.elevation_label.config(text=f"{self.elevations[frame]:.2f}°")
        self.azimuth_label.config(text=f"{self.azimuths[frame]:.2f}°")
        self.pitch_label.config(text=f"{90 - self.elevations[frame]:.2f}°")
        self.roll_label.config(text=f"{self.azimuths[frame]:.2f}°")
        
        if frame == len(self.times) - 1:
            self.animation.event_source.stop()
        
        return self.panel,

    def run_simulation(self):
        """Ejecuta la simulación completa"""
        try:
            self.clean_previous_animation()
            
            # Obtener parámetros de entrada
            date = self.date_entry.get_date()
            hour_start = int(self.hour_spin.get())
            duration = int(self.duration_spin.get())
            interval = int(self.interval_spin.get())
            
            start_time = TIMEZONE.localize(
                datetime.combine(date, datetime.min.time()) + timedelta(hours=hour_start))
            
            # Calcular trayectoria solar
            self.times, self.sun_vectors, self.elevations, self.azimuths = \
                self.calculate_sun_positions(start_time, duration, interval)
            
            # Configurar figura y ejes
            self.fig = plt.Figure(figsize=(10, 8), dpi=100)
            self.ax = self.fig.add_subplot(111, projection='3d')
            
            # Configurar canvas
            if hasattr(self, 'canvas') and self.canvas is not None:
                self.canvas.get_tk_widget().destroy()
            
            self.canvas = FigureCanvasTkAgg(self.fig, self.plot_frame)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Crear animación
            self.animation = FuncAnimation(
                self.fig, self.update_animation,
                frames=len(self.times),
                init_func=self.init_animation,
                interval=200,
                blit=False,
                repeat=False
            )
            
            self.canvas.draw()
            
        except Exception as e:
            print(f"Error en simulación: {str(e)}")
            import traceback
            traceback.print_exc()

    def clean_previous_animation(self):
        """Limpia recursos de animaciones anteriores"""
        if hasattr(self, 'animation') and self.animation:
            self.animation.event_source.stop()
        
        if hasattr(self, 'canvas') and self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
        
        if hasattr(self, 'fig') and self.fig:
            plt.close(self.fig)
            self.fig = None
        
        self.ax = None
        self.sun_path = None
        self.panel = None
        self.quiver = None
        self.sun_dot = None


if __name__ == "__main__":
    root = tk.Tk()
    app = SolarTrackerApp(root)
    root.mainloop()