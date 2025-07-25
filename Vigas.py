import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def calculate_reactions(L, loads):
    """Calcular reacciones en los apoyos para viga simplemente apoyada"""
    Ra = 0
    Rb = 0
    Ma = 0
    
    for load in loads:
        if load['type'] == 'point':
            P = load['magnitude']
            a = load['position']
            b = L - a
            Ra += P * b / L
            Rb += P * a / L
        elif load['type'] == 'distributed':
            w = load['magnitude']
            start = load['start']
            end = load['end']
            length = end - start
            total_load = w * length
            centroid = start + length / 2
            a = centroid
            b = L - a
            Ra += total_load * b / L
            Rb += total_load * a / L
        elif load['type'] == 'moment':
            M = load['magnitude']
            Ma += M
            Ra += M / L
            Rb -= M / L
    
    return Ra, Rb, Ma

def calculate_shear_moment(L, loads, Ra, Rb, n_points=1000):
    """Calcular cortante y momento a lo largo de la viga"""
    x = np.linspace(0, L, n_points)
    V = np.zeros(n_points)
    M = np.zeros(n_points)
    
    for i, xi in enumerate(x):
        # Cortante
        Vi = Ra
        for load in loads:
            if load['type'] == 'point' and xi > load['position']:
                Vi -= load['magnitude']
            elif load['type'] == 'distributed' and xi > load['start']:
                start = load['start']
                end = min(xi, load['end'])
                if end > start:
                    Vi -= load['magnitude'] * (end - start)
        V[i] = Vi
        
        # Momento
        Mi = Ra * xi
        for load in loads:
            if load['type'] == 'point' and xi > load['position']:
                Mi -= load['magnitude'] * (xi - load['position'])
            elif load['type'] == 'distributed' and xi > load['start']:
                start = load['start']
                end = min(xi, load['end'])
                if end > start:
                    length = end - start
                    centroid = start + length / 2
                    Mi -= load['magnitude'] * length * (xi - centroid)
            elif load['type'] == 'moment' and xi > load['position']:
                Mi -= load['magnitude']
        M[i] = Mi
    
    return x, V, M

def main():
    st.set_page_config(page_title="Diseño de Vigas - Momentos y Cortantes", layout="wide")
    
    st.title("🏗️ DISEÑO DE VIGAS - CÁLCULO DE MOMENTOS Y CORTANTES")
    st.markdown("---")
    
    # Sidebar para parámetros de la viga
    st.sidebar.header("Parámetros de la Viga")
    
    # Longitud de la viga
    L = st.sidebar.number_input("Longitud de la viga (m)", min_value=1.0, max_value=50.0, value=6.0, step=0.5)
    
    # Tipo de apoyo
    support_type = st.sidebar.selectbox("Tipo de apoyo", ["Simplemente apoyada", "Empotrada-libre", "Empotrada-empotrada"])
    
    if support_type != "Simplemente apoyada":
        st.warning("⚠️ Por ahora solo se soportan vigas simplemente apoyadas. Otras configuraciones estarán disponibles próximamente.")
        return
    
    st.sidebar.markdown("---")
    st.sidebar.header("Cargas Aplicadas")
    
    # Inicializar lista de cargas en session state
    if 'loads' not in st.session_state:
        st.session_state.loads = []
    
    # Selector de tipo de carga
    load_type = st.sidebar.selectbox("Tipo de carga", ["Puntual", "Distribuida", "Momento"])
    
    col1, col2 = st.sidebar.columns(2)
    
    if load_type == "Puntual":
        with col1:
            magnitude = st.number_input("Fuerza (kN)", value=10.0, step=1.0, key="point_mag")
        with col2:
            position = st.number_input("Posición (m)", min_value=0.0, max_value=L, value=L/2, step=0.1, key="point_pos")
        
        if st.sidebar.button("Agregar Carga Puntual"):
            st.session_state.loads.append({
                'type': 'point',
                'magnitude': magnitude,
                'position': position,
                'description': f"P = {magnitude} kN en x = {position} m"
            })
    
    elif load_type == "Distribuida":
        with col1:
            magnitude = st.number_input("Carga (kN/m)", value=5.0, step=0.5, key="dist_mag")
        with col2:
            start = st.number_input("Inicio (m)", min_value=0.0, max_value=L, value=0.0, step=0.1, key="dist_start")
        
        end = st.sidebar.number_input("Fin (m)", min_value=start, max_value=L, value=L, step=0.1, key="dist_end")
        
        if st.sidebar.button("Agregar Carga Distribuida"):
            st.session_state.loads.append({
                'type': 'distributed',
                'magnitude': magnitude,
                'start': start,
                'end': end,
                'description': f"w = {magnitude} kN/m de {start} a {end} m"
            })
    
    elif load_type == "Momento":
        with col1:
            magnitude = st.number_input("Momento (kN·m)", value=15.0, step=1.0, key="mom_mag")
        with col2:
            position = st.number_input("Posición (m)", min_value=0.0, max_value=L, value=L/2, step=0.1, key="mom_pos")
        
        if st.sidebar.button("Agregar Momento"):
            st.session_state.loads.append({
                'type': 'moment',
                'magnitude': magnitude,
                'position': position,
                'description': f"M = {magnitude} kN·m en x = {position} m"
            })
    
    # Mostrar cargas aplicadas
    if st.session_state.loads:
        st.sidebar.markdown("### Cargas Aplicadas:")
        for i, load in enumerate(st.session_state.loads):
            col1, col2 = st.sidebar.columns([3, 1])
            col1.write(f"{i+1}. {load['description']}")
            if col2.button("🗑️", key=f"delete_{i}"):
                st.session_state.loads.pop(i)
                st.rerun()
    
    if st.sidebar.button("Limpiar todas las cargas"):
        st.session_state.loads = []
        st.rerun()
    
    # Cálculos principales
    if st.session_state.loads:
        # Calcular reacciones
        Ra, Rb, Ma = calculate_reactions(L, st.session_state.loads)
        
        # Calcular cortante y momento
        x, V, M = calculate_shear_moment(L, st.session_state.loads, Ra, Rb)
        
        # Mostrar resultados
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Reacciones en los Apoyos")
            st.metric("Reacción en A (Ra)", f"{Ra:.2f} kN", delta=None)
            st.metric("Reacción en B (Rb)", f"{Rb:.2f} kN", delta=None)
            if abs(Ma) > 0.01:
                st.metric("Momento en A (Ma)", f"{Ma:.2f} kN·m", delta=None)
            
            # Verificación de equilibrio
            total_vertical = sum([load['magnitude'] if load['type'] == 'point' else 
                                load['magnitude'] * (load['end'] - load['start']) if load['type'] == 'distributed' else 0 
                                for load in st.session_state.loads])
            
            st.markdown("### ✅ Verificación de Equilibrio")
            st.write(f"ΣFy = {Ra + Rb - total_vertical:.3f} kN ≈ 0")
            
        with col2:
            st.subheader("📈 Valores Máximos")
            max_shear = max(abs(V))
            max_moment = max(abs(M))
            max_shear_pos = x[np.argmax(np.abs(V))]
            max_moment_pos = x[np.argmax(np.abs(M))]
            
            st.metric("Cortante Máximo", f"{max_shear:.2f} kN", delta=None)
            st.write(f"Posición: {max_shear_pos:.2f} m")
            st.metric("Momento Máximo", f"{max_moment:.2f} kN·m", delta=None)
            st.write(f"Posición: {max_moment_pos:.2f} m")
        
        # Gráficos
        st.markdown("---")
        st.subheader("📊 Diagramas de Cortante y Momento")
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # Diagrama de cargas
        ax1.set_xlim(0, L)
        ax1.axhline(y=0, color='black', linewidth=2)
        ax1.plot([0, L], [0, 0], 'ko-', markersize=8, linewidth=3)  # Viga
        
        # Dibujar apoyos
        ax1.plot(0, 0, '^', markersize=12, color='red', label='Apoyo A')
        ax1.plot(L, 0, '^', markersize=12, color='red', label='Apoyo B')
        
        # Dibujar cargas
        for load in st.session_state.loads:
            if load['type'] == 'point':
                ax1.arrow(load['position'], 2, 0, -1.5, head_width=L*0.02, head_length=0.2, fc='blue', ec='blue')
                ax1.text(load['position'], 2.5, f"{load['magnitude']} kN", ha='center', fontsize=10)
            elif load['type'] == 'distributed':
                x_dist = np.linspace(load['start'], load['end'], 20)
                y_dist = np.ones_like(x_dist) * 2
                ax1.fill_between(x_dist, 0, y_dist, alpha=0.3, color='green')
                for xi in x_dist[::3]:
                    ax1.arrow(xi, 2, 0, -1.5, head_width=L*0.01, head_length=0.1, fc='green', ec='green')
                ax1.text((load['start'] + load['end'])/2, 2.5, f"{load['magnitude']} kN/m", ha='center', fontsize=10)
        
        ax1.set_ylim(-1, 3)
        ax1.set_ylabel('Cargas')
        ax1.set_title('Esquema de Cargas')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Diagrama de cortante
        ax2.plot(x, V, 'b-', linewidth=2, label='Cortante')
        ax2.fill_between(x, 0, V, where=(V >= 0), alpha=0.3, color='blue', label='Cortante (+)')
        ax2.fill_between(x, 0, V, where=(V < 0), alpha=0.3, color='red', label='Cortante (-)')
        ax2.axhline(y=0, color='black', linewidth=1)
        ax2.set_ylabel('Cortante (kN)')
        ax2.set_title('Diagrama de Cortante')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Diagrama de momento
        ax3.plot(x, M, 'r-', linewidth=2, label='Momento')
        ax3.fill_between(x, 0, M, where=(M >= 0), alpha=0.3, color='orange', label='Momento (+)')
        ax3.fill_between(x, 0, M, where=(M < 0), alpha=0.3, color='purple', label='Momento (-)')
        ax3.axhline(y=0, color='black', linewidth=1)
        ax3.set_xlabel('Posición (m)')
        ax3.set_ylabel('Momento (kN·m)')
        ax3.set_title('Diagrama de Momento Flector')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Tabla de valores
        st.markdown("---")
        st.subheader("📋 Tabla de Valores")
        
        # Crear tabla con valores cada L/20 puntos
        n_table = 21
        x_table = np.linspace(0, L, n_table)
        V_table = np.interp(x_table, x, V)
        M_table = np.interp(x_table, x, M)
        
        df = pd.DataFrame({
            'Posición (m)': x_table,
            'Cortante (kN)': V_table,
            'Momento (kN·m)': M_table
        })
        
        df = df.round(3)
        st.dataframe(df, use_container_width=True)
        
        # Descargar resultados
        st.markdown("---")
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Descargar resultados (CSV)",
            data=csv,
            file_name=f"analisis_viga_{L}m.csv",
            mime="text/csv"
        )
    
    else:
        st.info("👆 Agrega cargas usando el panel lateral para comenzar el análisis.")
        
        # Mostrar ejemplo
        st.markdown("### 📖 Ejemplo de Uso:")
        st.markdown("""
        1. **Define la longitud** de la viga en el panel lateral
        2. **Agrega cargas** utilizando los controles:
           - **Carga Puntual**: Fuerza concentrada en un punto
           - **Carga Distribuida**: Carga uniforme en un tramo
           - **Momento**: Momento aplicado en un punto
        3. **Visualiza** los diagramas de cortante y momento
        4. **Analiza** los resultados y descarga los datos
        """)

if __name__ == "__main__":
    main()