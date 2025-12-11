import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Importa os modelos para execução
import ghi_mamdani
import ghi_sugeno

def garantir_pasta_predict():
    if not os.path.exists("predict"):
        os.makedirs("predict")

def calcular_z_mamdani(simulador, inputs_dict):
    """Wrapper para calcular saída do Mamdani (Scikit-Fuzzy)"""
    try:
        for key, value in inputs_dict.items():
            simulador.input[key] = value
        simulador.compute()
        return simulador.output['ghi']
    except:
        return 0.0

def calcular_z_sugeno(inputs_dict):
    """Wrapper para calcular saída do Sugeno (Função Pura)"""
    # Mapeia os nomes do dicionário para os argumentos da função sugeno
    return ghi_sugeno.avaliar_ghi_sugeno(
        h_sin=inputs_dict.get('hora_sin', 0),
        h_cos=inputs_dict.get('hora_cos', -1),
        nuvem=inputs_dict.get('tipo_nuvem', 0),
        temp=inputs_dict.get('temp_ar', 25)
    )

def gerar_superficie(tipo_modelo, x_nome, x_range, y_nome, y_range, fixos, titulo):
    """
    Gera e salva a superfície de controle 3D.
    
    Args:
        tipo_modelo (str): 'mamdani' ou 'sugeno'
        x_nome (str): Nome da variável no eixo X (ex: 'hora_cos')
        x_range (list): [min, max] do eixo X
        y_nome (str): Nome da variável no eixo Y (ex: 'tipo_nuvem')
        y_range (list): [min, max] do eixo Y
        fixos (dict): Valores fixos para as outras 2 variáveis
        titulo (str): Título do gráfico
    """
    
    print(f"Gerando gráfico 3D para {tipo_modelo}: {x_nome} vs {y_nome}...")

    # 1. Cria a malha (Grid)
    resolucao = 30
    x = np.linspace(x_range[0], x_range[1], resolucao)
    y = np.linspace(y_range[0], y_range[1], resolucao)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    # 2. Loop de Simulação
    for i in range(resolucao):
        for j in range(resolucao):
            # Monta o dicionário de inputs para este ponto
            inputs = fixos.copy()
            inputs[x_nome] = X[i, j]
            inputs[y_nome] = Y[i, j]

            # Calcula Z baseado no modelo
            if tipo_modelo == 'mamdani':
                Z[i, j] = calcular_z_mamdani(ghi_mamdani.simulador, inputs)
            elif tipo_modelo == 'sugeno':
                Z[i, j] = calcular_z_sugeno(inputs)

    # 3. Plotagem
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot da superfície com mapa de cores 'viridis' (padrão científico)
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k', lw=0.1, alpha=0.85)
    
    # Labels e Ajustes
    ax.set_xlabel(x_nome, fontsize=12, labelpad=10)
    ax.set_ylabel(y_nome, fontsize=12, labelpad=10)
    ax.set_zlabel('GHI (W/m²)', fontsize=12, labelpad=10)
    ax.set_title(f"{titulo} ({tipo_modelo.capitalize()})", fontsize=14, fontweight='bold')
    
    # Barra de cores
    cbar = fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1)
    cbar.set_label('Irradiação Solar (W/m²)')

    # Ajuste de ângulo de visão (Elevacao, Azimute) para ver melhor a curva
    ax.view_init(elev=30, azim=135)

    # 4. Salvar
    garantir_pasta_predict()
    filename = f"predict/surface_{tipo_modelo}_{x_nome}_vs_{y_nome}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Salvo em: {filename}")

if __name__ == "__main__":
    # --- CONFIGURAÇÃO DOS CENÁRIOS ---
    
    # CENÁRIO A: Elevação Solar (hora_cos) vs Nuvens (tipo_nuvem)
    # Fixamos Temperatura em 25°C e Hora Seno em 0 (Meio-dia/Indiferente)
    gerar_superficie(
        tipo_modelo='mamdani',
        x_nome='hora_cos', x_range=[-1, 1],
        y_nome='tipo_nuvem', y_range=[0, 10],
        fixos={'temp_ar': 25, 'hora_sin': 0},
        titulo="Impacto da Elevação Solar e Nuvens"
    )

    gerar_superficie(
        tipo_modelo='sugeno',
        x_nome='hora_cos', x_range=[-1, 1],
        y_nome='tipo_nuvem', y_range=[0, 10],
        fixos={'temp_ar': 25, 'hora_sin': 0},
        titulo="Impacto da Elevação Solar e Nuvens"
    )

    # CENÁRIO B: Ciclo Diário (hora_sin) vs Elevação (hora_cos) -> A "Banheira" do Sol
    # Fixamos Nuvens em 0 (Céu Limpo) e Temperatura em 25°C
    gerar_superficie(
        tipo_modelo='mamdani',
        x_nome='hora_sin', x_range=[-1, 1],
        y_nome='hora_cos', y_range=[-1, 1],
        fixos={'tipo_nuvem': 0, 'temp_ar': 25},
        titulo="Ciclo Diário em Céu Limpo"
    )
    
    gerar_superficie(
        tipo_modelo='sugeno',
        x_nome='hora_sin', x_range=[-1, 1],
        y_nome='hora_cos', y_range=[-1, 1],
        fixos={'tipo_nuvem': 0, 'temp_ar': 25},
        titulo="Ciclo Diário em Céu Limpo"
    )