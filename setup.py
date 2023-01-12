# Carrera sobre la cual se hace el grafo
# Puede ser cualquiera de:
# sistemas informatica agrimensura alimentos civil electricista electronica 
# industrial mecanica naval petroleo quimica sistemasviejo
CARRERA = 'informatica'

# Padron para los analisis particulares
PADRON = '100029'

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
