import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx

from config import PADRON, CARRERA, plan_estudios
from math import log, e, ceil
from numpy import linalg as LA


pd.set_option('mode.chained_assignment', None)
categories = {
    4: 0,
    5: 0,
    6: 1,
    7: 1,
    8: 2,
    9: 2,
    10: 2
}

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


def generar_subdf_materias_electivas(df):
    df_materias = pd.read_json(plan_estudios(CARRERA))
    df_alumnos = pd.merge(df, df_materias, left_on='materia_id', right_on="id")
    df_alumnos = df_alumnos[df_alumnos['categoria'] == 'Materias Electivas'][['Padron', 'materia_id', 'materia_nota']]

    return df_alumnos[['Padron', 'materia_id', 'materia_nota']].copy()

def curar_data(df):
    # Sacamos materias en final y a cursar
    df_alumnos = df[df['materia_nota'] >= 4]

    # Sacamos gente que no le pone la nota a su fiubamap y deja que se saco (casi) todos 4s directamente
    df_alumnos['mediana'] = df_alumnos.groupby('Padron')['materia_nota'].transform('median')
    df_alumnos = df_alumnos[df_alumnos['mediana'] > 5]

    df_alumnos['nota_categoria'] = df_alumnos['materia_nota'].apply(lambda x: categories[x])

    # Juntamos el grafo con si mismo para tener la similiritud entre cada par de padrones
    df_simil = pd.merge(df_alumnos, df_alumnos, on=['materia_id', 'nota_categoria'])
    df_simil = df_simil[df_simil['Padron_x'] != df_simil['Padron_y']]
    df_simil = df_simil.reset_index()

    df_simil = df_simil[['materia_id', 'nota_categoria', 'Padron_x', 'materia_nota_x', 'Padron_y', 'materia_nota_y']]

    # Unificar aristas ( con el objetivo de no tener pocos nodos y millones de aristas )
    df_simil_agg = df_simil.groupby(['Padron_x', 'Padron_y']).agg(cant_materias_similares=('materia_id', 'count'))
    df_simil_agg = df_simil_agg.reset_index()

    df_simil_agg['Padron_min'] = df_simil_agg[['Padron_x', 'Padron_y']].min(axis=1)
    df_simil_agg['Padron_max'] = df_simil_agg[['Padron_x', 'Padron_y']].max(axis=1)
    df_simil_agg = df_simil_agg.drop_duplicates(['Padron_min', 'Padron_max']).reset_index()

    # ToDo: fijarse si elevar a un K acá
    df_simil_agg['inv_cant_materias_similares'] = df_simil_agg['cant_materias_similares'].max() - df_simil_agg['cant_materias_similares'] + 1
    return df_simil_agg[['Padron_x', 'Padron_y', 'cant_materias_similares', 'inv_cant_materias_similares']]


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

def plot_diametro(G):
    shortest_paths = nx.shortest_path(G, source=random.choice(list(G.nodes)))
    target = max(shortest_paths, key=lambda i: len(shortest_paths[i]))
    diameter = shortest_paths[target]
    diameter_edges = list(zip(diameter, diameter[1:]))

    pos = nx.spiral_layout(G)
    nx.draw(G, pos=pos, with_labels=True, width=0.005, node_size=5, font_size=6)
    nx.draw_networkx_nodes(G, pos, nodelist=diameter, node_size=10, node_color='r')
    nx.draw_networkx_edges(G, edge_color='r', width=4.0, edgelist=diameter_edges, pos=pos, node_size=30)
