# config.py
import plotly.graph_objects as go

# Paleta de colores moderna en tonos oscuros (neutros, apagados)
COLORS = {
  "background": "#0d1117",  # Fondo oscuro
  "card": "#161b22",        # Tarjetas ligeramente más claras
  "dark": "#c9d1d9",        # Texto principal en blanco/gris claro
  "light": "#30363d",       # Bordes
  "primary": "#58a6ff",     # Azul suave
  "secondary": "#238636",   # Verde suave
  "accent": "#a371f7",      # Púrpura suave
  "success": "#3fb950",     # Verde éxito
  "warning": "#d29922",     # Amarillo/naranja
  "danger": "#f85149",      # Rojo suave
  "gray": "#8b949e",        # Texto secundario
}

# Configuración de plotly para tema oscuro
PLOTLY_TEMPLATE = go.layout.Template(
  layout=go.Layout(
    paper_bgcolor=COLORS["card"],
    plot_bgcolor=COLORS["card"],
    font={"color": COLORS["dark"]},
    xaxis={"gridcolor": COLORS["light"], "linecolor": COLORS["light"]},
    yaxis={"gridcolor": COLORS["light"], "linecolor": COLORS["light"]},
  )
)

# Información de sensores
SENSOR_INFO = {
  1: {"name": "T2", "description": "Total temperature at fan inlet", "unit": "°R", "critical": True},
  2: {"name": "T24", "description": "Total temperature at LPC outlet", "unit": "°R", "critical": True},
  3: {"name": "T30", "description": "Total temperature at HPC outlet", "unit": "°R", "critical": True},
  4: {"name": "T50", "description": "Total temperature at LPT outlet", "unit": "°R", "critical": True},
  5: {"name": "P2", "description": "Static pressure at fan inlet", "unit": "psia", "critical": False},
  6: {"name": "P15", "description": "Total pressure in bypass duct", "unit": "psia", "critical": False},
  7: {"name": "P30", "description": "Total pressure at HPC outlet", "unit": "psia", "critical": True},
  8: {"name": "Nf", "description": "Physical fan speed", "unit": "rpm", "critical": True},
  9: {"name": "Nc", "description": "Physical core speed", "unit": "rpm", "critical": True},
  10: {"name": "epr", "description": "Engine pressure ratio (P50/P2)", "unit": "—", "critical": True},
  11: {"name": "Ps30", "description": "Static pressure at HPC outlet", "unit": "psia", "critical": True},
  12: {"name": "phi", "description": "Ratio of fuel flow to Ps30", "unit": "pps/psi", "critical": False},
  13: {"name": "NRf", "description": "Corrected fan speed", "unit": "rpm", "critical": False},
  14: {"name": "NRc", "description": "Corrected core speed", "unit": "rpm", "critical": False},
  15: {"name": "BPR", "description": "Bypass ratio", "unit": "—", "critical": False},
  16: {"name": "farB", "description": "Burner fuel-air ratio", "unit": "—", "critical": False},
  17: {"name": "htBleed", "description": "Bleed enthalpy", "unit": "—", "critical": False},
  18: {"name": "Nf_dmd", "description": "Required fan speed", "unit": "rpm", "critical": False},
  19: {"name": "PCNfR_dmd", "description": "Required corrected fan speed", "unit": "rpm", "critical": False},
  20: {"name": "W31", "description": "High-pressure turbine coolant bleed", "unit": "lbm/s", "critical": True},
  21: {"name": "W32", "description": "Low-pressure turbine coolant bleed", "unit": "lbm/s", "critical": True}
}