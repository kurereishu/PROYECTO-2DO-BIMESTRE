import tkinter as tk 
from tkinter import ttk, filedialog, messagebox
from tkcalendar import DateEntry
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import art3d
from datetime import datetime, timedelta
from pytz import timezone
from pysolar.solar import get_altitude, get_azimuth
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Configuración geográfica
latitude = -0.2105367
longitude = -78.491614
timezone_local = timezone("America/Guayaquil")

class SolarTrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Seguidor Solar 2-DOF - Control Matemático")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Estilo mejorado
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabelframe.Label', font=('Arial', 12, 'bold'))
        style.configure('Header.TLabel', font=('Arial', 10, 'bold'))
        style.configure('Action.TButton', font=('Arial', 10, 'bold'))
        
        self.create_widgets()
        
        # Variables
        self.animation = None
        self.fig = None
        self.ax = None
        self.canvas = None
        self.sun_vectors = None
        self.times = None
        self.current_frame = 0
        self.panel = None
        self.elevations = []
        self.azimuths = []
        self.pitch_angles = []
        self.roll_angles = []
        self.panel_width = 0.5
        self.panel_height = 0.8
        self.sun_distance = 1.0
        self.reference_vector = np.array([0, 1, 0])
        self.angle_text = None
        
        # Elementos gráficos de ángulos
        self.elevation_arc = None
        self.azimuth_arc = None
        self.elevation_angle_label = None
        self.azimuth_angle_label = None
        self.legend_text = None

    def create_widgets(self):
        # === Marco principal izquierdo ===
        left_frame = ttk.Frame(self.root)
        left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        # === Marco de configuración con mejor diseño ===
        config_frame = ttk.LabelFrame(left_frame, text="⚙️ Configuración del Sistema", 
                                      padding=15, style='Title.TLabelframe')
        config_frame.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        # Configuración en grid más organizado
        ttk.Label(config_frame, text="📅 Fecha de simulación:", style='Header.TLabel').grid(
            row=0, column=0, sticky="w", pady=8)
        self.date_entry = DateEntry(config_frame, date_pattern='yyyy-mm-dd', 
                                   font=('Arial', 10))
        self.date_entry.grid(row=0, column=1, padx=10, pady=8, sticky="w")
        
        ttk.Label(config_frame, text="🕐 Hora de inicio:", style='Header.TLabel').grid(
            row=1, column=0, sticky="w", pady=8)
        self.hour_spin = ttk.Spinbox(config_frame, from_=0, to=23, width=10, 
                                    font=('Arial', 10))
        self.hour_spin.set(6)
        self.hour_spin.grid(row=1, column=1, padx=10, pady=8, sticky="w")
        
        ttk.Label(config_frame, text="⏱️ Duración (horas):", style='Header.TLabel').grid(
            row=2, column=0, sticky="w", pady=8)
        self.duration_spin = ttk.Spinbox(config_frame, from_=1, to=12, width=10, 
                                        font=('Arial', 10))
        self.duration_spin.set(12)
        self.duration_spin.grid(row=2, column=1, padx=10, pady=8, sticky="w")
        
        ttk.Label(config_frame, text="⏲️ Intervalo (minutos):", style='Header.TLabel').grid(
            row=3, column=0, sticky="w", pady=8)
        self.interval_spin = ttk.Spinbox(config_frame, from_=1, to=60, width=10, 
                                        font=('Arial', 10))
        self.interval_spin.set(15)
        self.interval_spin.grid(row=3, column=1, padx=10, pady=8, sticky="w")
        
        # Botones de acción
        buttons_frame = ttk.Frame(config_frame)
        buttons_frame.grid(row=4, column=0, columnspan=2, pady=15)
        
        self.run_button = ttk.Button(buttons_frame, text="▶️ Calcular y Animar", 
                                    command=self.run_simulation, style='Action.TButton')
        self.run_button.pack(side="left", padx=5)
        
        self.save_button = ttk.Button(buttons_frame, text="💾 Guardar Reporte", 
                                     command=self.save_report, style='Action.TButton')
        self.save_button.pack(side="left", padx=5)
        
        # === Marco de ángulos mejorado ===
        angles_frame = ttk.LabelFrame(left_frame, text="📐 Ángulos Calculados en Tiempo Real", 
                                      padding=15, style='Title.TLabelframe')
        angles_frame.grid(row=1, column=0, padx=5, pady=10, sticky="ew")
        
        # Grid para los ángulos
        angles_data = [
            ("🔺 Elevación Solar (θ):", "elevation_label", "0.00°", "Ángulo vertical del sol"),
            ("🧭 Azimuth Solar (α):", "azimuth_label", "0.00°", "Ángulo horizontal del sol"),
            ("📐 Pitch del Panel:", "pitch_label", "0.00°", "Inclinación vertical"),
            ("🔄 Roll del Panel:", "roll_label", "0.00°", "Rotación horizontal")
        ]
        
        for i, (label, attr, default, tooltip) in enumerate(angles_data):
            ttk.Label(angles_frame, text=label, style='Header.TLabel').grid(
                row=i, column=0, sticky="w", pady=5)
            label_widget = ttk.Label(angles_frame, text=default, 
                                   font=('Arial', 11, 'bold'), foreground='#2E86C1')
            label_widget.grid(row=i, column=1, sticky="e", padx=10, pady=5)
            setattr(self, attr, label_widget)
        
        # === Marco de información del sistema ===
        info_frame = ttk.LabelFrame(left_frame, text="ℹ️ Información del Sistema", 
                                    padding=15, style='Title.TLabelframe')
        info_frame.grid(row=2, column=0, padx=5, pady=10, sticky="ew")
        
        info_text = f"""📍 Ubicación: Quito, Ecuador
🌐 Latitud: {latitude:.6f}°
🌍 Longitud: {longitude:.6f}°
⏰ Zona Horaria: {timezone_local}"""
        
        ttk.Label(info_frame, text=info_text, font=('Arial', 9)).pack()
        
        # === Marco de visualización 3D ===
        self.plot_frame = ttk.LabelFrame(self.root, text="🌅 Visualización 3D - Trayectoria Solar", 
                                        padding=10, style='Title.TLabelframe')
        self.plot_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        # === Controles de animación mejorados ===
        self.controls_frame = ttk.LabelFrame(self.root, text="🎮 Controles de Animación", 
                                           padding=10)
        self.controls_frame.grid(row=1, column=1, pady=10, sticky="ew")
        
        self.playing = False
        
        control_buttons = ttk.Frame(self.controls_frame)
        control_buttons.pack(side="top", pady=5)
        
        self.play_button = ttk.Button(control_buttons, text="▶️ Play", 
                                     command=self.toggle_play, width=8)
        self.play_button.pack(side="left", padx=3)
        
        self.back_button = ttk.Button(control_buttons, text="⏪", 
                                     command=self.step_back, width=4)
        self.back_button.pack(side="left", padx=3)
        
        self.forward_button = ttk.Button(control_buttons, text="⏩", 
                                        command=self.step_forward, width=4)
        self.forward_button.pack(side="left", padx=3)
        
        self.reset_button = ttk.Button(control_buttons, text="🔄 Reset", 
                                      command=self.reiniciar_animacion, width=8)
        self.reset_button.pack(side="left", padx=3)
        
        # Slider y tiempo
        slider_frame = ttk.Frame(self.controls_frame)
        slider_frame.pack(side="top", fill="x", padx=10, pady=5)
        
        ttk.Label(slider_frame, text="Tiempo:").pack(side="left", padx=(0, 10))
        self.slider = ttk.Scale(slider_frame, from_=0, to=0, orient="horizontal", 
                               command=self.slider_moved)
        self.slider.pack(side="left", fill="x", expand=True, padx=5)
        
        self.time_label = ttk.Label(slider_frame, text="--:--", 
                                   font=('Arial', 10, 'bold'))
        self.time_label.pack(side="left", padx=(10, 0))
        
        # Configurar grid
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

    def save_report(self):
        """Guarda un reporte completo con tabla de datos"""
        if not hasattr(self, 'times') or self.times is None:
            messagebox.showwarning("Advertencia", "Primero debe ejecutar la simulación")
            return
            
        if not PIL_AVAILABLE:
            messagebox.showerror("Error", "PIL (Pillow) no está instalado.\nInstale con: pip install pillow")
            return
            
        # Seleccionar ubicación de guardado
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")],
            title="Guardar Reporte del Seguidor Solar"
        )
        
        if not filename:
            return
            
        try:
            self.generate_report_image(filename)
            messagebox.showinfo("Éxito", f"Reporte guardado exitosamente en:\n{filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Error al guardar el reporte:\n{str(e)}")

    def generate_report_image(self, filename):
        """Genera una imagen con el reporte completo"""
        # Crear imagen base
        img_width, img_height = 1200, 1600
        img = Image.new('RGB', (img_width, img_height), 'white')
        draw = ImageDraw.Draw(img)
        
        try:
            title_font = ImageFont.truetype("arial.ttf", 24)
            header_font = ImageFont.truetype("arial.ttf", 16)
            data_font = ImageFont.truetype("arial.ttf", 12)
        except:
            title_font = ImageFont.load_default()
            header_font = ImageFont.load_default()
            data_font = ImageFont.load_default()
        
        y_pos = 30
        
        # Título
        title = "REPORTE DE SEGUIDOR SOLAR 2-DOF"
        draw.text((img_width//2 - 200, y_pos), title, fill='black', font=title_font)
        y_pos += 60
        
        # Información de configuración
        config_info = [
            f"Fecha de simulacion: {self.date_entry.get()}",
            f"Hora de inicio: {self.hour_spin.get()}:00",
            f"Duracion: {self.duration_spin.get()} horas",
            f"Intervalo: {self.interval_spin.get()} minutos",
            f"Ubicacion: Quito, Ecuador ({latitude:.4f}°, {longitude:.4f}°)",
            f"Zona horaria: {timezone_local}",
            f"Total de mediciones: {len(self.times)}"
        ]
        
        for info in config_info:
            draw.text((50, y_pos), info, fill='black', font=header_font)
            y_pos += 25
        
        y_pos += 20
        
        # Encabezado de tabla
        draw.text((50, y_pos), "DATOS DE SEGUIMIENTO SOLAR POR HORA", fill='black', font=header_font)
        y_pos += 40
        
        # Encabezados de columna
        headers = ["Hora", "Elevacion (°)", "Azimuth (°)", "Pitch (°)", "Roll (°)", "Observaciones"]
        col_widths = [80, 100, 100, 80, 80, 200]
        x_positions = [50]
        for width in col_widths[:-1]:
            x_positions.append(x_positions[-1] + width)
        
        # Dibujar línea de encabezado
        for i, (header, x_pos) in enumerate(zip(headers, x_positions)):
            draw.text((x_pos, y_pos), header, fill='black', font=header_font)
        
        y_pos += 30
        
        # Línea separadora
        draw.line([(40, y_pos), (img_width-40, y_pos)], fill='black', width=2)
        y_pos += 15
        
        # Datos por hora (filtrar para mostrar solo datos cada hora)
        hourly_data = self.get_hourly_data()
        
        for i, data in enumerate(hourly_data):
            if y_pos > img_height - 100:  # Si nos quedamos sin espacio
                break
                
            time_str = data['time'].strftime("%H:%M")
            elevation = f"{data['elevation']:.1f}"
            azimuth = f"{data['azimuth']:.1f}"
            pitch = f"{data['pitch']:.1f}"
            roll = f"{data['roll']:.1f}"
            
            # Observaciones basadas en los valores
            obs = self.get_observation(data['elevation'], data['azimuth'])
            
            row_data = [time_str, elevation, azimuth, pitch, roll, obs]
            
            # Alternar color de fila para mejor legibilidad
            if i % 2 == 1:
                draw.rectangle([(40, y_pos-5), (img_width-40, y_pos+20)], 
                              fill='#f0f0f0', outline=None)
            
            for j, (text, x_pos) in enumerate(zip(row_data, x_positions)):
                if j == len(row_data) - 1:  # Observaciones
                    # Texto más pequeño para observaciones
                    draw.text((x_pos, y_pos), text[:30] + "..." if len(text) > 30 else text, 
                             fill='black', font=data_font)
                else:
                    draw.text((x_pos, y_pos), text, fill='black', font=data_font)
            
            y_pos += 25
        
        # Resumen estadístico
        y_pos += 30
        draw.text((50, y_pos), "RESUMEN ESTADISTICO", fill='black', font=header_font)
        y_pos += 30
        
        stats = self.calculate_statistics()
        for stat in stats:
            draw.text((50, y_pos), stat, fill='black', font=data_font)
            y_pos += 20
        
        # Guardar imagen
        img.save(filename, quality=95)

    def get_hourly_data(self):
        """Obtiene datos filtrados por hora"""
        hourly_data = []
        current_hour = None
        
        for i, time in enumerate(self.times):
            if current_hour != time.hour:
                current_hour = time.hour
                hourly_data.append({
                    'time': time,
                    'elevation': self.elevations[i],
                    'azimuth': self.azimuths[i],
                    'pitch': self.pitch_angles[i],
                    'roll': self.roll_angles[i]
                })
        
        return hourly_data

    def get_observation(self, elevation, azimuth):
        """Genera observaciones basadas en los ángulos"""
        if elevation < 10:
            return "Sol bajo en horizonte"
        elif elevation > 70:
            return "Sol en cenit - maxima radiacion"
        elif 30 <= elevation <= 70:
            return "Condiciones optimas"
        elif azimuth > 270 or azimuth < 90:
            return "Sol hacia el este"
        elif 90 <= azimuth <= 270:
            return "Sol hacia el oeste"
        else:
            return "Seguimiento normal"

    def calculate_statistics(self):
        """Calcula estadísticas del seguimiento"""
        stats = []
        if self.elevations:
            max_elev_idx = self.elevations.index(max(self.elevations))
            stats.extend([
                f"• Elevacion maxima: {max(self.elevations):.1f}° a las {self.times[max_elev_idx].strftime('%H:%M')}",
                f"• Elevacion minima: {min(self.elevations):.1f}°",
                f"• Elevacion promedio: {np.mean(self.elevations):.1f}°",
                f"• Rango de azimuth: {min(self.azimuths):.1f}° a {max(self.azimuths):.1f}°",
                f"• Tiempo de seguimiento: {len(self.times)} mediciones",
                f"• Eficiencia estimada: {self.calculate_efficiency():.1f}%"
            ])
        return stats

    def calculate_efficiency(self):
        """Calcula eficiencia estimada basada en ángulos de elevación"""
        if not self.elevations:
            return 0
        
        # Eficiencia basada en elevaciones > 10°
        good_angles = [e for e in self.elevations if e > 10]
        return (len(good_angles) / len(self.elevations)) * 100

    def clean_previous_animation(self):
        if self.animation:
            self.animation.event_source.stop()
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        if self.fig:
            plt.close(self.fig)
        self.animation = None
        self.canvas = None
        self.fig = None
        self.ax = None
        self.panel = None
        self.angle_text = None
        self.elevation_arc = None
        self.azimuth_arc = None
        self.elevation_angle_label = None
        self.azimuth_angle_label = None
        self.legend_text = None

    def calculate_sun_position(self, date, hour_start, duration_hours, time_step_minutes):
        start_time = timezone_local.localize(datetime.combine(date, datetime.min.time()) + timedelta(hours=hour_start))
        times, elevations, azimuths = [], [], []
        pitch_angles, roll_angles = [], []
        for i in range(0, duration_hours * 60, time_step_minutes):
            t = start_time + timedelta(minutes=i)
            elevation = get_altitude(latitude, longitude, t)
            azimuth = get_azimuth(latitude, longitude, t)
            pitch = 90 - elevation
            roll = azimuth
            times.append(t)
            elevations.append(elevation)
            azimuths.append(azimuth)
            pitch_angles.append(pitch)
            roll_angles.append(roll)
        elev_rad = np.radians(elevations)
        azim_rad = np.radians(azimuths)
        sun_vectors = np.array([
            np.cos(elev_rad) * np.sin(azim_rad),
            np.cos(elev_rad) * np.cos(azim_rad),
            np.sin(elev_rad)
        ]).T
        return times, sun_vectors, elevations, azimuths, pitch_angles, roll_angles

    def create_panel_vertices(self, normal_vector):
        normal = normal_vector / np.linalg.norm(normal_vector)
        if np.allclose(np.abs(normal), [0, 0, 1]):
            u = np.array([1, 0, 0])
        else:
            u = np.cross(self.reference_vector, normal)
            u = u / np.linalg.norm(u)
        v = np.cross(normal, u)
        v = v / np.linalg.norm(v)
        hw = self.panel_width / 2
        hh = self.panel_height / 2
        return [[hh * u + hw * v, -hh * u + hw * v, -hh * u - hw * v, hh * u - hw * v]]

    def create_angle_arc(self, start_vec, end_vec, center=np.array([0,0,0]), radius=0.3, num_points=20):
        start_vec = start_vec / np.linalg.norm(start_vec)
        end_vec = end_vec / np.linalg.norm(end_vec)
        normal = np.cross(start_vec, end_vec)
        if np.linalg.norm(normal) == 0:
            normal = np.array([0,0,1])
        else:
            normal = normal / np.linalg.norm(normal)
        u = start_vec
        v = np.cross(normal, u)
        v = v / np.linalg.norm(v)
        angle_start = 0
        angle_end = np.arccos(np.clip(np.dot(start_vec, end_vec), -1.0, 1.0))
        angles = np.linspace(angle_start, angle_end, num_points)
        arc = np.array([center + radius * (np.cos(a) * u + np.sin(a) * v) for a in angles])
        return arc

    def init_animation(self):
        sun_path = self.sun_vectors * self.sun_distance
        self.ax.plot(sun_path[:, 0], sun_path[:, 1], sun_path[:, 2],
                     'y-', alpha=0.5, marker='o', markersize=3, label="Trayectoria solar")
        panel_verts = self.create_panel_vertices(self.sun_vectors[0])
        self.panel = Poly3DCollection(panel_verts, color='green', alpha=0.8) 
        self.ax.add_collection3d(self.panel)
        self.ax.set_xlim([-1.2, 1.2])
        self.ax.set_ylim([-1.2, 1.2])
        self.ax.set_zlim([0, 1.5])
        self.ax.set_xlabel("Este-Oeste")
        self.ax.set_ylabel("Norte-Sur")
        self.ax.set_zlabel("Altura")

        self.legend_text = self.ax.text2D(
            0.05, 0.05,
            "● = Elevacion (rojo)\n● = Azimuth (azul)",
            transform=self.ax.transAxes, fontsize=9, color='black',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
        )

        self.ax.legend()
        return self.panel,

    def update_animation(self, frame):
        self.current_frame = frame
        normal = self.sun_vectors[frame]
        elevation = self.elevations[frame]
        azimuth = self.azimuths[frame]

        # Actualizar panel
        self.panel.set_verts(self.create_panel_vertices(normal))

        # Limpiar arcos anteriores
        if self.elevation_arc is not None:
            self.elevation_arc.remove()
            self.elevation_arc = None
        if self.azimuth_arc is not None:
            self.azimuth_arc.remove()
            self.azimuth_arc = None
        if self.elevation_angle_label is not None:
            self.elevation_angle_label.remove()
            self.elevation_angle_label = None
        if self.azimuth_angle_label is not None:
            self.azimuth_angle_label.remove()
            self.azimuth_angle_label = None

        # Limpiar otros elementos
        for artist in self.ax.lines[1:]:
            artist.remove()
        for artist in self.ax.collections[1:]:
            if artist != self.panel:
                artist.remove()

        # Dibujar vector solar y sol
        self.ax.quiver(0, 0, 0, normal[0], normal[1], normal[2],
                       color='orange', length=1.2, arrow_length_ratio=0.1)
        self.ax.scatter(normal[0] * self.sun_distance, normal[1] * self.sun_distance,
                        normal[2] * self.sun_distance,
                        color='yellow', s=200, edgecolor='orange', zorder=10)

        # Arco de elevación (rojo)
        proj_horizontal = np.array([normal[0], normal[1], 0])
        if np.linalg.norm(proj_horizontal) > 0.01 and elevation > 5:
            arc_elev = self.create_angle_arc(proj_horizontal, normal, radius=0.3, num_points=15)
            self.elevation_arc = art3d.Line3DCollection([arc_elev], colors=['red'], linewidths=2)
            self.ax.add_collection3d(self.elevation_arc)
            mid_point = arc_elev[len(arc_elev)//2]
            self.elevation_angle_label = self.ax.text(
                mid_point[0], mid_point[1], mid_point[2],
                f"{elevation:.1f}°", fontsize=8, color='red', ha='center', va='center'
            )

        # Arco de azimuth (azul)
        north = np.array([0, 1, 0])
        proj_norm = np.array([normal[0], normal[1], 0])
        if np.linalg.norm(proj_norm) > 0.01 and abs(azimuth) > 5:
            arc_azim = self.create_angle_arc(north, proj_norm, radius=0.2, num_points=15)
            self.azimuth_arc = art3d.Line3DCollection([arc_azim], colors=['blue'], linewidths=2)
            self.ax.add_collection3d(self.azimuth_arc)
            mid_point = arc_azim[len(arc_azim)//2]
            self.azimuth_angle_label = self.ax.text(
                mid_point[0], mid_point[1], mid_point[2],
                f"{azimuth:.1f}°", fontsize=8, color='blue', ha='center', va='center'
            )

        # Texto de ángulos (esquina superior)
        if self.angle_text:
            self.angle_text.remove()
        self.angle_text = self.ax.text2D(
            0.05, 0.95,
            f"Elevacion: {elevation:.2f}°\n"
            f"Azimuth: {azimuth:.2f}°\n"
            f"Pitch: {self.pitch_angles[frame]:.2f}°\n"
            f"Roll: {self.roll_angles[frame]:.2f}°",
            transform=self.ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.8)
        )

        self.ax.set_title(f"Posicion solar a las {self.times[frame].strftime('%H:%M')}")
        self.elevation_label.config(text=f"{elevation:.2f}°")
        self.azimuth_label.config(text=f"{azimuth:.2f}°")
        self.pitch_label.config(text=f"{self.pitch_angles[frame]:.2f}°")
        self.roll_label.config(text=f"{self.roll_angles[frame]:.2f}°")
        self.slider.set(frame)
        self.time_label.config(text=self.times[frame].strftime("%H:%M"))

        self.canvas.draw_idle()

        # Detener animación al final
        if frame == len(self.times) - 1:
            if self.animation and self.animation.event_source:
                self.animation.event_source.stop()
            self.playing = False
            self.play_button.config(text="▶️ Play")

        return self.panel,

    def run_simulation(self):
        try:
            self.clean_previous_animation()
            date = self.date_entry.get_date()
            hour_start = int(self.hour_spin.get())
            duration = int(self.duration_spin.get())
            interval = int(self.interval_spin.get())
            (self.times, self.sun_vectors, self.elevations,
             self.azimuths, self.pitch_angles, self.roll_angles) = self.calculate_sun_position(
                date, hour_start, duration, interval)

            self.fig = plt.Figure(figsize=(12, 8), dpi=100)
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.canvas = FigureCanvasTkAgg(self.fig, self.plot_frame)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            self.animation = FuncAnimation(
                self.fig, self.update_animation,
                frames=len(self.times),
                init_func=self.init_animation,
                interval=200, blit=False, repeat=False
            )

            self.slider.config(to=len(self.times) - 1)
            self.slider.set(0)
            self.current_frame = 0
            self.playing = False
            self.play_button.config(text="▶️ Play")
            self.time_label.config(text="--:--")
            self.canvas.draw_idle()

        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Error en la simulación:\n{str(e)}")

    def toggle_play(self):
        if not self.animation or not self.animation.event_source:
            return
        if self.playing:
            self.animation.event_source.stop()
            self.play_button.config(text="▶️ Play")
        else:
            # Reiniciar secuencia y avanzar hasta el frame actual
            self.animation.frame_seq = self.animation.new_frame_seq()
            for _ in range(self.current_frame):
                try:
                    next(self.animation.frame_seq)
                except StopIteration:
                    break
            self.animation.event_source.start()
            self.play_button.config(text="⏸️ Pause")
        self.playing = not self.playing

    def step_forward(self):
        if self.sun_vectors is None: return
        if self.current_frame < len(self.sun_vectors) - 1:
            self.current_frame += 1
            self.update_animation(self.current_frame)
            # Sincronizar animación
            if self.animation and self.animation.event_source:
                self.animation.frame_seq = self.animation.new_frame_seq()
                for _ in range(self.current_frame):
                    try:
                        next(self.animation.frame_seq)
                    except StopIteration:
                        break
            self.canvas.draw_idle()

    def step_back(self):
        if self.sun_vectors is None: return
        if self.current_frame > 0:
            self.current_frame -= 1
            self.update_animation(self.current_frame)
            if self.animation and self.animation.event_source:
                self.animation.frame_seq = self.animation.new_frame_seq()
                for _ in range(self.current_frame):
                    try:
                        next(self.animation.frame_seq)
                    except StopIteration:
                        break
            self.canvas.draw_idle()

    def slider_moved(self, val):
        if self.sun_vectors is None: return
        frame = int(float(val))
        if frame != self.current_frame:
            self.current_frame = frame
            self.update_animation(frame)
            if self.animation and self.animation.event_source:
                self.animation.frame_seq = self.animation.new_frame_seq()
                for _ in range(self.current_frame):
                    try:
                        next(self.animation.frame_seq)
                    except StopIteration:
                        break
            self.canvas.draw_idle()

    def reiniciar_animacion(self):
        if self.sun_vectors is None: return
        self.current_frame = 0
        self.update_animation(0)
        if self.animation and self.animation.event_source:
            self.animation.event_source.stop()
            self.animation.frame_seq = self.animation.new_frame_seq()
        self.playing = False
        self.play_button.config(text="▶️ Play")
        self.slider.set(0)
        self.time_label.config(text=self.times[0].strftime("%H:%M"))

if __name__ == "__main__":
    root = tk.Tk()
    app = SolarTrackerApp(root)
    root.mainloop()