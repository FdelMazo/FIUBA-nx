import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx

from config import PADRON, CARRERA
from math import log, e, ceil
from numpy import linalg as LA

plt.rcParams['figure.figsize'] = (30,10)
pd.set_option('mode.chained_assignment', None)

DIFFERENT_WALKS = { # incluye bucles
    2: 2,
    3: 5,
    4: 15,
    5: 52,
    6: 203,
    7: 877,
    8: 4140,
    9: 21147,
    10: 115975,
    11: 678570,
    12: 4213597
}

LENGHTS = [5, 6, 7]

# Imprime las estadísticas que nos interesan de un grafo
def stats(G):
    print(G)
    print(f"""
  El diámetro de la red: {nx.diameter(G)}
  El grado promedio de la red: {sum([n[1] for n in G.degree()]) / len(G):.2f}
  Puentes globales: {list(nx.bridges(G))}
""")

# Plotea un grafo en dos gráficos side by side
def plot(G, edge_width=0.005):
    fig, axes = plt.subplots(nrows=1, ncols=2)
    ax = axes.flatten()
    # Un plot circular, para darnos una idea de cuan completo es (a ojo)
    nx.draw_networkx(G, pos=nx.circular_layout(G), width=edge_width, node_size=50, with_labels=False, ax=ax[0])

    # Un plot que nos muestre el diametro en rojo
    shortest_paths = nx.shortest_path(G, source=random.choice(list(G.nodes)))
    target = max(shortest_paths, key=lambda i: len(shortest_paths[i]))
    diameter = shortest_paths[target]
    diameter_edges = list(zip(diameter, diameter[1:]))

    pos = nx.spiral_layout(G)
    nx.draw(G, pos=pos, with_labels=True, width=edge_width, node_size=5, font_size=6)
    nx.draw_networkx_nodes(G, pos, nodelist=diameter, node_size=5, node_color='r')
    nx.draw_networkx_edges(G, edge_color='r', width=2.0, edgelist=diameter_edges, pos=pos, node_size=30)

# Plotea cada comunidad en un color distinto
def plot_communities(G, louvain):
    draw_nodes = {}
    colors = random.sample(list(mcolors.CSS4_COLORS), len(louvain))
    for louvaincommunity, color in zip(louvain, colors):
        draw_nodes.update({n: color for n in louvaincommunity})

    plt.title(f"{len(louvain)} Louvain Communities")
    nx.draw_networkx(G,
                     nodelist=draw_nodes.keys(),
                     node_color=list(draw_nodes.values()),
                     width=0.005,
                     pos=nx.kamada_kawai_layout(G),
                     font_size=10)

def generar_subdf_materias_electivas(df):
    df_materias = pd.read_json(plan_estudios(CARRERA))
    df_alumnos = pd.merge(df, df_materias, left_on='materia_id', right_on="id")
    df_alumnos = df_alumnos[df_alumnos['categoria'] == 'Materias Electivas'][['Padron', 'materia_id', 'materia_nota']]

    return df_alumnos[['Padron', 'materia_id', 'materia_nota']].copy()

def ln(num):
    return log(num, e)


def _cant_annonymous_walks(length, error=0.1, delta=0.01):
    nu = DIFFERENT_WALKS[length]
    return ceil((ln(2**nu - 2) - ln(delta)) * (2 / (error**2)))


def _camino_a_clave(camino):
    return "-".join(map(lambda v: str(v), camino))


def _annon_enum_rec(pasos_restantes, mapeo, camino=[], vs_en_camino=0, admite_bucles=False):
    if pasos_restantes == 0:
        mapeo[_camino_a_clave(camino)] = len(mapeo)
        return
    nuevo = vs_en_camino + 1
    ultimo = camino[-1] if len(camino) > 0 else None
    for i in range(1, nuevo + 1):
        if i == ultimo and not admite_bucles:
            continue
        camino.append(i)
        vs_en_este_camino = vs_en_camino + (1 if i == nuevo else 0)
        _annon_enum_rec(pasos_restantes - 1, mapeo, camino, vs_en_este_camino)
        camino.pop()


def _enumerar_anonymous_walks(length):
    mapeo = {}
    _annon_enum_rec(length, mapeo)
    return mapeo


def _random_walk(grafo, length):
    v = random.choice(list(grafo.nodes())) 
    camino = [v] 
    while len(camino) < length: 
        neighhbors = list(grafo.neighbors(v))
        if not neighhbors: return []
        v = random.choice(neighhbors)
        camino.append(v)
    return camino


def _anonymize_walk(camino):
    translate = {}
    camino_trans = []
    for v in camino:
        if v not in translate:
            translate[v] = len(translate) + 1
        camino_trans.append(translate[v])
    return camino_trans


def anoymous_walks(grafo):
    '''
    :param grafo: Grafo a calcularle el embedding por anonymous_walk
    :return: diccionario: para cada largo N, su determinado arreglo de probabilidad
    '''
    AN = {}
    for n in LENGHTS:
        cantidad = _cant_annonymous_walks(n)
        mapeo = _enumerar_anonymous_walks(n)
        contadores = [0] * len(mapeo)
        for i in range(cantidad):
            camino = _random_walk(grafo, n) 
            contadores[mapeo[_camino_a_clave(_anonymize_walk(camino))]] += 1
        vector = np.array(contadores)
        AN[n] = vector / LA.norm(vector)
    return AN

names=['Original', 'Erdös-Rényi', 'Preferential Attachment']

def plot_anoymous_walks(graphs):
    colors = ['r', 'b', 'g']
    fig = plt.figure("Anonymous walks", figsize=(10, 10))
    axgrid = fig.add_gridspec(3, 3)
    i,j = 0, 0
    for l in LENGHTS:
        ax1 = fig.add_subplot(axgrid[j, :])
        for g in graphs:
            ax1.plot(sorted(g[l]), colors[i], marker="o", label= names[i])
            i += 1
        ax1.set_title("Representación de AN largo " + str(l))
        ax1.set_ylabel("Probabilidad")
        ax1.set_xlabel("n")
        plt.legend()
        i = 0
        j += 1

def plot_distribucion_grados(grafos):
    fig = plt.figure("Distribución de grados", figsize=(10, 10))
    axgrid = fig.add_gridspec(4, 8)

    i = 0
    for j, g in enumerate(grafos):
        degree_sequence = sorted((d for n, d in g.degree()), reverse=True)
        ax = fig.add_subplot(axgrid[0, i:i+2])

        ax.bar(*np.unique(degree_sequence, return_counts=True))
        ax.set_title("Histograma " + names[j])
        ax.set_xlabel("Distribución")
        ax.set_ylabel("# de Nodos")
        i += 3    

# Traer el plan de estudios del FIUBA-Map (y rezar que nunca cambie tanto como para que se rompa la interfaz)
def plan_estudios(carrera):
    # Hardcodear los planes, por si algun dia el fiuba map sube los planes 2020
    PLANES = {
        'informatica': 'informatica-1986',
        'agrimensura': 'agrimensura-2006',
        'alimentos': 'alimentos-2000',
        'civil': 'civil-2009',
        'electricista': 'electricista-2009',
        'electronica': 'electronica-2009',
        'industrial': 'industrial-2011',
        'mecanica': 'mecanica-1986',
        'naval': 'naval-1986',
        'petroleo': 'petroleo-2015',
        'quimica': 'quimica-1986',
        'sistemas': 'sistemas-2014',
        'sistemasviejo': 'sistemas-1986'
    }
    return f'https://raw.githubusercontent.com/fdelmazo/FIUBA-Map/master/src/data/{PLANES[carrera]}.json'
